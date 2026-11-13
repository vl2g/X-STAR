"""End-to-end evaluation on a synthetic layout: full gallery, explicit IDs, strict baseline."""

import json

import numpy as np
import pytest
import torch
from fixtures import make_layout

from xstar.checkpoints import build_model, load_checkpoint
from xstar.config import PathsConfig
from xstar.constants import QUERY_SETS
from xstar.data import DataError, ImageTokens, OcrBank, PackedQueries, load_global_queries
from xstar.evaluation import evaluate
from xstar.metrics import relevance_indices, relevant_ranks
from xstar.retrieval import build_bank
from xstar.scoring import maxsim_reference


@pytest.fixture
def layout(tmp_path):
    return make_layout(tmp_path)


def test_complete_run_scores_full_gallery_by_id(layout):
    paths = PathsConfig.load(layout["paths"])
    result = evaluate(paths, layout["checkpoint"], log=lambda *_: None)
    assert result["complete_benchmark"] and result["gallery_size"] == 5
    assert set(result["colite_clip"]["per_query_set"]) == set(QUERY_SETS)
    assert result["clip_baseline"]["mode"] == "global_query_vs_global_image"

    # Recompute english_query independently with the reference MaxSim loop.
    model, K = build_model(load_checkpoint(layout["checkpoint"]))
    images = ImageTokens(paths.image_tokens("test"), paths.split_list("test"))
    X, Xm = build_bank(model, images, images.ids, OcrBank(paths.ocr_dir("test"), K))
    pq = PackedQueries(*paths.packed_queries("test", "english_query"))
    assert pq.ids != images.ids                     # rows are reversed on purpose
    Q, Qm = pq.single_caption()
    S = maxsim_reference(torch.from_numpy(Q), torch.from_numpy(Qm), X, Xm)
    ranks, _ = relevant_ranks(S, relevance_indices(pq.ids, images.ids))
    expected = {f"R@{k}": float((ranks <= k).mean() * 100) for k in (1, 5, 10)}
    assert result["colite_clip"]["per_query_set"]["english_query"] == pytest.approx(expected)
    assert result["query_counts"]["english_query"]["queries"] == 5


def test_partial_runs_are_labelled(layout):
    paths = PathsConfig.load(layout["paths"])
    r = evaluate(paths, layout["checkpoint"], query_sets=["english_query"], max_queries=2, log=lambda *_: None)
    assert not r["complete_benchmark"]
    assert len(r["partial_reasons"]) == 2
    assert r["gallery_size"] == 5 and r["query_counts"]["english_query"]["queries"] == 2


@pytest.mark.parametrize("victim", ["global_query", "global_ids", "image_global"])
def test_missing_baseline_features_stop_the_run(layout, victim):
    paths = PathsConfig.load(layout["paths"])
    vec, ids, _ = paths.global_queries("test", "hindi_to_english")
    target = {"global_query": vec, "global_ids": ids, "image_global": paths.image_global("test")}[victim]
    target.unlink()
    with pytest.raises(DataError, match="missing evaluation inputs"):
        evaluate(paths, layout["checkpoint"], log=lambda *_: None)
    with pytest.raises(DataError, match="never used as a substitute"):
        if victim == "image_global":
            from xstar.data import load_image_global
            load_image_global(target, 5)
        else:
            load_global_queries(vec, ids)


def test_misaligned_global_ids_are_rejected(layout):
    paths = PathsConfig.load(layout["paths"])
    _, ids, _ = paths.global_queries("test", "odia_to_english")
    lines = ids.read_text().split()
    ids.write_text("\n".join(lines[::-1]) + "\n")
    with pytest.raises(DataError, match="differ from packed-query IDs"):
        evaluate(paths, layout["checkpoint"], query_sets=["odia_to_english"], log=lambda *_: None)


def test_missing_ocr_file_is_an_error(layout):
    paths = PathsConfig.load(layout["paths"])
    next(paths.ocr_dir("test").glob("*.npz")).unlink()
    with pytest.raises(DataError, match="no OCR file"):
        evaluate(paths, layout["checkpoint"], query_sets=["english_query"], log=lambda *_: None)


def test_null_visual_ablation_changes_bank(layout):
    paths = PathsConfig.load(layout["paths"])
    model, K = build_model(load_checkpoint(layout["checkpoint"]))
    images = ImageTokens(paths.image_tokens("test"), paths.split_list("test"))
    ocr = OcrBank(paths.ocr_dir("test"), K)
    X, _ = build_bank(model, images, images.ids, ocr)
    Xn, _ = build_bank(model, images, images.ids, ocr, null_visual=True)
    assert not np.allclose(X.numpy(), Xn.numpy())


# ---------------------------------------------------------------------------
# Coverage validation against the manifest split.
# ---------------------------------------------------------------------------

def _rewrite_query_set(paths, qs, select):
    """Rewrite a query set's packed and global files consistently with rows ``select(ids)``."""
    tok_p, mask_p, ids_p = paths.packed_queries("test", qs)
    g_p, gids_p, str_p = paths.global_queries("test", qs)
    ids = ids_p.read_text().split()
    rows = select(list(range(len(ids))))
    tok, mask, g = np.load(tok_p), np.load(mask_p), np.load(g_p)
    strings = str_p.read_text().splitlines()
    np.save(tok_p, tok[rows]); np.save(mask_p, mask[rows]); np.save(g_p, g[rows])
    new_ids = [ids[r] for r in rows]
    ids_p.write_text("\n".join(new_ids) + "\n")
    gids_p.write_text("\n".join(new_ids) + "\n")
    str_p.write_text("\n".join(strings[r] for r in rows) + "\n")
    return new_ids


def test_truncated_query_set_is_rejected_by_default(layout):
    """The reviewer's reproduction: English packed/global files shortened consistently to two rows."""
    paths = PathsConfig.load(layout["paths"])
    _rewrite_query_set(paths, "english_query", lambda r: r[:2])
    with pytest.raises(DataError, match="english_query: query cache does not cover the test split"):
        evaluate(paths, layout["checkpoint"], log=lambda *_: None)


def test_truncated_query_set_allowed_only_as_explicit_partial(layout):
    paths = PathsConfig.load(layout["paths"])
    _rewrite_query_set(paths, "english_query", lambda r: r[:2])
    r = evaluate(paths, layout["checkpoint"], allow_incomplete_queries=True, log=lambda *_: None)
    assert not r["complete_benchmark"]
    assert any("english_query: 2 of 5" in reason for reason in r["partial_reasons"])
    assert r["gallery_size"] == 5 and r["query_counts"]["english_query"]["queries"] == 2
    assert r["coverage"]["query_sets"]["english_query"]["missing_ids"] == 3


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_duplicate_query_ids_are_always_rejected(layout, allow_incomplete):
    paths = PathsConfig.load(layout["paths"])
    _rewrite_query_set(paths, "tamil_to_english", lambda r: r[:-1] + [r[0]])     # same length, one ID twice
    with pytest.raises(DataError, match="tamil_to_english: query IDs are not a set"):
        evaluate(paths, layout["checkpoint"], allow_incomplete_queries=allow_incomplete, log=lambda *_: None)


def test_query_ids_outside_the_split_are_rejected(layout):
    paths = PathsConfig.load(layout["paths"])
    tok_p, mask_p, ids_p = paths.packed_queries("test", "hindi_to_english")
    _, gids_p, _ = paths.global_queries("test", "hindi_to_english")
    ids = ids_p.read_text().split()
    ids[0] = "A_image_1"                                   # a train image
    for p in (ids_p, gids_p):
        p.write_text("\n".join(ids) + "\n")
    with pytest.raises(DataError, match="extra 1"):
        evaluate(paths, layout["checkpoint"], allow_incomplete_queries=True, log=lambda *_: None)


def _edit_manifest(layout, fn):
    data = json.loads(layout["manifest"].read_text())
    fn(data["data"])
    layout["manifest"].write_text(json.dumps(data))


def test_gallery_missing_a_manifest_image_is_rejected(layout):
    def add_record(records):
        rec = dict(next(r for r in records if r["split"] == "test"))
        rec.update(image_id="B_image_99", filename="B/image_99.jpg")
        records.append(rec)
    _edit_manifest(layout, add_record)
    paths = PathsConfig.load(layout["paths"])
    with pytest.raises(DataError, match=r"does not match the manifest's test split.*missing 1"):
        evaluate(paths, layout["checkpoint"], log=lambda *_: None)


def test_gallery_with_an_image_outside_the_split_is_rejected(layout):
    _edit_manifest(layout, lambda records: records.__setitem__(
        slice(None), [r for r in records if r["image_id"] != "B_image_8"]))
    paths = PathsConfig.load(layout["paths"])
    with pytest.raises(DataError, match=r"does not match the manifest's test split.*extra 1"):
        evaluate(paths, layout["checkpoint"], query_sets=["english_query"], log=lambda *_: None)


def test_shuffled_but_aligned_gallery_gives_identical_results(tmp_path):
    base = make_layout(tmp_path / "a")
    shuffled = make_layout(tmp_path / "b")
    paths_b = PathsConfig.load(shuffled["paths"])
    order = [3, 0, 4, 2, 1]
    lst = paths_b.split_list("test")
    names = lst.read_text().split()
    lst.write_text("\n".join(names[i] for i in order) + "\n")
    for p in (paths_b.image_tokens("test"), paths_b.image_global("test")):
        np.save(p, np.load(p)[order])
    ra = evaluate(PathsConfig.load(base["paths"]), base["checkpoint"], log=lambda *_: None)
    rb = evaluate(paths_b, shuffled["checkpoint"], log=lambda *_: None)
    assert rb["complete_benchmark"] and rb["coverage"]["gallery"]["ordered_ids_sha256"] != \
        ra["coverage"]["gallery"]["ordered_ids_sha256"]
    for key in ("colite_clip", "clip_baseline"):
        assert rb[key]["per_query_set"] == ra[key]["per_query_set"]


def test_complete_fixture_reports_full_coverage(layout):
    r = evaluate(PathsConfig.load(layout["paths"]), layout["checkpoint"], log=lambda *_: None)
    assert r["complete_benchmark"] and r["partial_reasons"] == []
    assert r["coverage"]["expected_split_ids_from_manifest"] == 5
    assert r["queries_evaluated"] == 5 * 13
    assert all(c["missing_ids"] == c["extra_ids"] == c["duplicated_ids"] == 0
               for c in r["coverage"]["query_sets"].values())


# ---------------------------------------------------------------------------
# Tie policy: equal scores keep the order of the split list (gallery order).
# ---------------------------------------------------------------------------

def _make_duplicate(paths, src_id, dst_id):
    """Give gallery image ``dst_id`` exactly the features of ``src_id`` (an identical image)."""
    import shutil
    ids = ImageTokens(paths.image_tokens("test"), paths.split_list("test")).ids
    a, b = ids.index(src_id), ids.index(dst_id)
    for p in (paths.image_tokens("test"), paths.image_global("test")):
        arr = np.load(p)
        arr[b] = arr[a]
        np.save(p, arr)
    shutil.copyfile(paths.ocr_dir("test") / f"{src_id}.npz", paths.ocr_dir("test") / f"{dst_id}.npz")


def _stable_positions(paths, checkpoint, qs):
    """Per query: 1-based position of every gallery ID under a stable descending sort (list order on ties)."""
    model, K = build_model(load_checkpoint(checkpoint))
    images = ImageTokens(paths.image_tokens("test"), paths.split_list("test"))
    X, Xm = build_bank(model, images, images.ids, OcrBank(paths.ocr_dir("test"), K))
    pq = PackedQueries(*paths.packed_queries("test", qs))
    Q, Qm = pq.single_caption()
    S = maxsim_reference(torch.from_numpy(Q), torch.from_numpy(Qm), X, Xm)
    order = torch.sort(S, dim=1, descending=True, stable=True).indices.tolist()
    return {qid: {images.ids[j]: pos + 1 for pos, j in enumerate(order[i])} for i, qid in enumerate(pq.ids)}


@pytest.mark.parametrize("swap", [False, True])
def test_identical_gallery_images_tie_in_split_list_order(layout, swap):
    paths = PathsConfig.load(layout["paths"])
    _make_duplicate(paths, "B_image_8", "B_image_9")
    if swap:                                     # reverse the two duplicates in the gallery order
        lst = paths.split_list("test")
        names = lst.read_text().split()
        i, j = names.index("B/image_8.jpg"), names.index("B/image_9.jpg")
        names[i], names[j] = names[j], names[i]
        lst.write_text("\n".join(names) + "\n")
        for p in (paths.image_tokens("test"), paths.image_global("test")):
            arr = np.load(p)
            arr[[i, j]] = arr[[j, i]]
            np.save(p, arr)
    r = evaluate(paths, layout["checkpoint"], query_sets=["english_query"], log=lambda *_: None)
    assert r["query_counts"]["english_query"]["colite_tied_queries"] == 2
    assert r["query_counts"]["english_query"]["baseline_tied_queries"] == 2
    pos = _stable_positions(paths, layout["checkpoint"], "english_query")
    first, second = ("B_image_9", "B_image_8") if swap else ("B_image_8", "B_image_9")
    for qid in pos:                              # for every query, the later copy in list order ranks right after
        assert pos[qid][second] == pos[qid][first] + 1
    ranks = {qid: p[qid] for qid, p in pos.items()}
    expected = {f"R@{k}": 100.0 * sum(v <= k for v in ranks.values()) / len(ranks) for k in (1, 5, 10)}
    assert r["colite_clip"]["per_query_set"]["english_query"] == pytest.approx(expected)
