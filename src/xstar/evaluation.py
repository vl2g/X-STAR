"""Full-gallery benchmark evaluation of CoLITE-CLIP and the CLIP global baseline.

Differences from the original multi-language evaluator (main-result notebook
cell 12), all intentional:

* The CLIP baseline always uses global query and global image embeddings. If
  either is missing the run stops; token or patch averages are never used.
* Every query set is scored against the same full split gallery. Relevance is
  resolved by image ID, not by assuming query row i matches gallery row i.
* Coverage is derived from the manifest: the gallery must contain exactly the
  selected split's image IDs, and every evaluated query set exactly one query
  per expected image (no duplicates, missing or extra IDs), with packed and
  global query IDs in the same order. All inputs are validated before the
  checkpoint is loaded and the gallery bank is built.
* A complete run requires all 13 query sets. ``query_sets`` subsets,
  ``max_queries`` and ``allow_incomplete_queries`` (query caches covering only
  part of the split) are explicit partial runs; the gallery is never reduced.
* Missing OCR files are errors instead of empty OCR.
* Tied scores are ordered deterministically (``metrics.TIE_POLICY``).
"""

from collections import Counter
from hashlib import sha256

import torch

from xstar.checkpoints import build_model, checkpoint_identity, load_checkpoint
from xstar.constants import QUERY_SET_LANGUAGE, QUERY_SETS
from xstar.data import DataError, ImageTokens, Manifest, OcrBank, PackedQueries, load_global_queries, load_image_global
from xstar.metrics import TIE_POLICY, macro_average, recall_at_k, relevance_indices, relevant_ranks
from xstar.retrieval import build_bank
from xstar.scoring import clip_global_scores, maxsim_scores
from xstar.utils import sha256_file

KS = (1, 5, 10)


def resolve_inputs(paths, split, query_sets):
    """Check every required file up front and report all missing ones together."""
    missing = []
    required = [paths.image_tokens(split), paths.split_list(split), paths.image_global(split), paths.manifest]
    for qs in query_sets:
        required += list(paths.packed_queries(split, qs)) + list(paths.global_queries(split, qs)[:2])
    for p in required:
        if not p.is_file():
            missing.append(str(p))
    if missing:
        raise DataError("missing evaluation inputs:\n  " + "\n  ".join(missing))


def expected_split_ids(manifest, split):
    """Image IDs of one split according to the manifest (each must be unique)."""
    ids = [r["image_id"] for r in manifest.split_records(split)]
    if not ids:
        raise DataError(f"{manifest.path}: no records for split {split!r}")
    dup = sorted(i for i, c in Counter(ids).items() if c > 1)
    if dup:
        raise DataError(f"{manifest.path}: duplicate image IDs in split {split!r}, e.g. {dup[:5]}")
    return set(ids)


def id_coverage(ids, expected):
    """Compare an ID list with the expected set: rows, duplicates, missing and extra IDs."""
    counts = Counter(ids)
    dup = sorted(i for i, c in counts.items() if c > 1)
    missing = sorted(expected - counts.keys())
    extra = sorted(counts.keys() - expected)
    return {"rows": len(ids), "unique_ids": len(counts), "expected_ids": len(expected),
            "duplicated_ids": len(dup), "missing_ids": len(missing), "extra_ids": len(extra),
            "examples": {"duplicated": dup[:5], "missing": missing[:5], "extra": extra[:5]}}


def _describe(cov):
    ex = cov["examples"]
    return (f"{cov['rows']} rows, {cov['unique_ids']} unique of {cov['expected_ids']} expected; "
            f"duplicated {cov['duplicated_ids']} {ex['duplicated']}, missing {cov['missing_ids']} {ex['missing']}, "
            f"extra {cov['extra_ids']} {ex['extra']}")


def validate_inputs(paths, split, query_sets, max_queries=None, allow_incomplete_queries=False):
    """Validate gallery and query coverage against the manifest before any model work.

    Returns ``(images, image_global, sets, coverage, partial_reasons)`` where
    ``sets`` maps each query set to its loaded packed queries and global
    embeddings.
    """
    resolve_inputs(paths, split, query_sets)
    partial_reasons = []
    if set(query_sets) != set(QUERY_SETS):
        partial_reasons.append(f"{len(query_sets)} of {len(QUERY_SETS)} query sets requested")
    if max_queries is not None:
        partial_reasons.append(f"first {max_queries} queries per set requested")

    expected = expected_split_ids(Manifest.load(paths.manifest), split)
    images = ImageTokens(paths.image_tokens(split), paths.split_list(split))
    gallery_cov = id_coverage(images.ids, expected)
    if gallery_cov["duplicated_ids"] or gallery_cov["missing_ids"] or gallery_cov["extra_ids"]:
        raise DataError(f"gallery {paths.split_list(split)} does not match the manifest's {split} split: "
                        + _describe(gallery_cov))
    image_global = load_image_global(paths.image_global(split), len(images.ids))

    sets, set_cov = {}, {}
    for qs in query_sets:
        packed = PackedQueries(*paths.packed_queries(split, qs))
        g_path, g_ids_path, _ = paths.global_queries(split, qs)
        g, g_ids = load_global_queries(g_path, g_ids_path)
        if g_ids != packed.ids:
            raise DataError(f"{qs}: global-query IDs ({g_ids_path}) differ from packed-query IDs "
                            f"({packed.paths[2]}) in content or order")
        cov = id_coverage(packed.ids, expected)
        if cov["duplicated_ids"] or cov["extra_ids"]:
            raise DataError(f"{qs}: query IDs are not a set of {split} images: " + _describe(cov))
        if cov["missing_ids"]:
            if not allow_incomplete_queries:
                raise DataError(f"{qs}: query cache does not cover the {split} split: " + _describe(cov)
                                + ". Use --allow-incomplete-queries for an explicitly partial run.")
            partial_reasons.append(f"{qs}: {cov['unique_ids']} of {cov['expected_ids']} expected queries present")
        sets[qs] = (packed, g, g_path, g_ids_path)
        set_cov[qs] = cov
    gallery_sha = sha256("\n".join(images.ids).encode("utf-8")).hexdigest()
    coverage = {"expected_split_ids_from_manifest": len(expected),
                "gallery": {**gallery_cov, "matches_manifest_split": True, "ordered_ids_sha256": gallery_sha},
                "query_sets": set_cov}
    return images, image_global, sets, coverage, partial_reasons


def evaluate(paths, checkpoint_path, split="test", query_sets=QUERY_SETS, ocr_variant=None, device="cpu",
             bank_batch=256, query_chunk=32, gallery_chunk=128, max_queries=None, null_visual=False,
             allow_incomplete_queries=False, log=print):
    query_sets = list(query_sets)
    unknown = [q for q in query_sets if q not in QUERY_SETS]
    if unknown:
        raise ValueError(f"unknown query sets: {unknown}")
    if len(set(query_sets)) != len(query_sets):
        raise ValueError("query sets listed more than once")
    images, image_global, sets, coverage, partial_reasons = validate_inputs(
        paths, split, query_sets, max_queries, allow_incomplete_queries)
    gallery_ids = images.ids

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    ckpt = load_checkpoint(checkpoint_path)
    model, K = build_model(ckpt, device=device)
    variant = ocr_variant or paths.ocr_variant
    ocr = OcrBank(paths.ocr_dir(split, variant), K)
    ocr.check_ids(gallery_ids)
    log(f"Inputs validated: gallery {len(gallery_ids)} images, {len(query_sets)} query sets"
        + (f" (partial: {'; '.join(partial_reasons)})" if partial_reasons else " (complete)"))
    log(f"Fusing {len(gallery_ids)} gallery images (K={K}, null_visual={null_visual}) ...")
    X, X_mask = build_bank(model, images, gallery_ids, ocr, batch_size=bank_batch, device=device,
                           null_visual=null_visual)
    X, X_mask = X.to(device), X_mask.to(device)

    per_set, baseline_per_set, counts, inputs = {}, {}, {}, {}
    for qs in query_sets:
        packed, g, g_path, g_ids_path = sets[qs]
        n = len(packed) if max_queries is None else min(max_queries, len(packed))
        ids = packed.ids[:n]
        relevant = relevance_indices(ids, gallery_ids)
        Q, Qm = packed.single_caption(slice(0, n))
        S = maxsim_scores(torch.from_numpy(Q).to(device), torch.from_numpy(Qm).to(device), X, X_mask,
                          query_chunk=query_chunk, gallery_chunk=gallery_chunk)
        ranks, ties = relevant_ranks(S, relevant)
        Sb = clip_global_scores(g[:n], image_global)
        ranks_b, ties_b = relevant_ranks(Sb, relevant)
        per_set[qs] = recall_at_k(ranks, KS)
        baseline_per_set[qs] = recall_at_k(ranks_b, KS)
        counts[qs] = {"queries": n, "queries_available": len(packed), "language": QUERY_SET_LANGUAGE[qs],
                      "colite_tied_queries": ties, "baseline_tied_queries": ties_b}
        inputs[qs] = {"packed_ids": str(packed.paths[2]), "packed_ids_sha256": sha256_file(packed.paths[2]),
                      "global_queries": str(g_path), "global_ids_sha256": sha256_file(g_ids_path)}
        log(f"[{qs}] n={n} CoLITE-CLIP {fmt(per_set[qs])} | CLIP {fmt(baseline_per_set[qs])}")

    return {
        "complete_benchmark": not partial_reasons,
        "partial_reasons": partial_reasons,
        "split": split,
        "gallery_size": len(gallery_ids),
        "query_sets": query_sets,
        "queries_evaluated": sum(c["queries"] for c in counts.values()),
        "coverage": coverage,
        "query_counts": counts,
        "colite_clip": {"per_query_set": per_set, "macro_average": macro_average(per_set, KS)},
        "clip_baseline": {"mode": "global_query_vs_global_image", "per_query_set": baseline_per_set,
                          "macro_average": macro_average(baseline_per_set, KS)},
        "settings": {"ocr_variant": variant, "ocr_max_lines": K, "null_visual_ablation": null_visual,
                     "precision": "float32", "device": device, "tie_policy": TIE_POLICY,
                     "tf32": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "float32_matmul_precision": torch.get_float32_matmul_precision()},
                     "torch_num_threads": torch.get_num_threads(),
                     "query_chunk": query_chunk, "gallery_chunk": gallery_chunk, "bank_batch": bank_batch,
                     "recall_ks": list(KS), "tau_mode": model.tau_mode, "stored_tau": float(model.tau)},
        "checkpoint": {**checkpoint_identity(checkpoint_path), "epoch": ckpt.get("epoch"),
                       "config": ckpt.get("config"), "metrics": ckpt.get("metrics")},
        "inputs": {"manifest": str(paths.manifest), "manifest_sha256": sha256_file(paths.manifest),
                   "split_list": str(paths.split_list(split)), "split_list_sha256": sha256_file(paths.split_list(split)),
                   "image_tokens": str(paths.image_tokens(split)), "image_global": str(paths.image_global(split)),
                   "ocr_dir": str(paths.ocr_dir(split, variant)), "query_sets": inputs},
    }


def fmt(m):
    return " / ".join(f"{m[f'R@{k}']:.2f}" for k in KS)
