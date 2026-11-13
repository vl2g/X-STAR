import json
import os
import pickle

import numpy as np
import pytest
from fixtures import make_layout

from xstar.config import PathsConfig
from xstar.data import (
    DataError,
    ImageTokens,
    Manifest,
    PackedQueries,
    TextCapsCaptionShards,
    load_ocr_embeddings,
    normalize_ocr_lines,
    pad_or_truncate,
    read_ocr_lines,
    sanitize,
    split_order,
    write_ocr_file,
)


def test_text_normalisation_matches_source():
    assert sanitize("  Lokpriya\tGopinath \n airport ") == "Lokpriya Gopinath airport"
    assert normalize_ocr_lines(["  a  b ", "", "   ", "c"]) == ["a b", "c"]


def test_legacy_ocr_file_numeric_read_without_pickle(tmp_path):
    legacy = tmp_path / "A_image_1.npz"
    E = np.eye(3, 512, dtype=np.float32)
    np.savez_compressed(legacy, E=E, tokens=np.array(["x", "y", "z"], dtype=object), scores=np.zeros(3, np.float32))
    with np.load(legacy, allow_pickle=False) as z, pytest.raises(ValueError):
        z["tokens"]                                       # object arrays need pickle
    np.testing.assert_array_equal(load_ocr_embeddings(legacy), E)
    with pytest.raises(DataError, match="allow_legacy_tokens"):
        read_ocr_lines(legacy)
    assert read_ocr_lines(legacy, allow_legacy_tokens=True) == ["x", "y", "z"]


def test_restricted_unpickler_blocks_arbitrary_objects(tmp_path):
    evil = tmp_path / "evil.npz"
    payload = pickle.dumps(np.array([os.system], dtype=object))  # would import os.system
    import zipfile
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("E.npy", _npy_bytes(np.zeros((1, 512), np.float32)))
        header = _npy_header_bytes(np.empty(1, dtype=object))
        zf.writestr("tokens.npy", header + payload)
    with pytest.raises(pickle.UnpicklingError):
        read_ocr_lines(evil, allow_legacy_tokens=True)


def _npy_bytes(arr):
    import io
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _npy_header_bytes(arr):
    import io
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, np.lib.format.header_data_from_array_1_0(arr))
    return buf.getvalue()


def test_new_ocr_format_roundtrip_and_no_overwrite(tmp_path):
    p = tmp_path / "A_image_2.npz"
    E = np.random.default_rng(0).standard_normal((2, 512)).astype(np.float32)
    write_ocr_file(p, E, ["first line", "second"])
    np.testing.assert_array_equal(load_ocr_embeddings(p), E)
    assert read_ocr_lines(p) == ["first line", "second"]
    with np.load(p, allow_pickle=False) as z:
        assert str(z["format"]) == "xstar-ocr-lines-v1"
    with pytest.raises(FileExistsError):
        write_ocr_file(p, E, ["first line", "second"])
    empty = tmp_path / "A_image_3.npz"
    write_ocr_file(empty, np.zeros((0, 512), np.float32), [])
    assert load_ocr_embeddings(empty).shape == (0, 512) and read_ocr_lines(empty) == []


def test_pad_or_truncate():
    E = np.ones((5, 512), np.float32)
    O, m = pad_or_truncate(E, 3)
    assert O.shape == (3, 512) and m.all()
    O, m = pad_or_truncate(E[:1], 3)
    assert m.tolist() == [True, False, False] and not O[1:].any()


def test_shape_and_order_checks(tmp_path):
    lay = make_layout(tmp_path)
    paths = PathsConfig.load(lay["paths"])
    images = ImageTokens(paths.image_tokens("train"), paths.split_list("train"))
    assert images.ids == ["A_image_1", "A_image_2", "B_image_3", "N_image_4"]
    with pytest.raises(DataError, match="no image tokens"):
        images.rows_for(["Z_image_0"])
    lst = paths.split_list("val")
    lst.write_text(lst.read_text() + "A/image_99.jpg\n")
    with pytest.raises(DataError, match="2 rows but 3 images"):
        ImageTokens(paths.image_tokens("val"), lst)
    tok, mask, ids = paths.packed_queries("train", "english_query")
    ids.write_text("A_image_1\n")
    with pytest.raises(DataError, match="misaligned"):
        PackedQueries(tok, mask, ids)


def test_manifest_validation_and_split_order(tmp_path):
    lay = make_layout(tmp_path)
    m = Manifest.load(lay["manifest"])
    assert m.validate() == []
    assert split_order(m, "test") == sorted(split_order(m, "test"))
    bad = tmp_path / "bad_list.txt"
    bad.write_text("A/image_7.jpg\n")
    with pytest.raises(DataError, match="does not match"):
        split_order(m, "test", bad)
    data = json.loads(lay["manifest"].read_text())
    data["data"][1]["image_id"] = data["data"][0]["image_id"]
    data["data"][2]["hindi_to_english"] = "   "
    m2 = Manifest(data["data"])
    problems = " | ".join(m2.validate())
    assert "duplicate image_id" in problems and "empty 'hindi_to_english'" in problems


def test_textcaps_shards_resolve_foreign_absolute_paths(tmp_path):
    rng = np.random.default_rng(0)
    shards = []
    for sid, rows in enumerate((2, 1)):
        np.save(tmp_path / f"emb_{sid}.npy", rng.standard_normal((rows, 5, 77, 512)).astype(np.float32))
        np.save(tmp_path / f"mask_{sid}.npy", np.ones((rows, 5, 77), bool))
        shards.append({"sid": sid, "emb": f"/elsewhere/emb_{sid}.npy", "mask": f"/elsewhere/mask_{sid}.npy", "rows": rows})
    (tmp_path / "index.json").write_text(json.dumps({"total_rows": 3, "shards": shards}))
    np.savez(tmp_path / "rowmap.npz", row2shard_ord=np.array([0, 0, 1], np.uint16), row2pos=np.array([0, 1, 0], np.uint32))
    (tmp_path / "ids.txt").write_text("a\nb\nc\n")
    idx = TextCapsCaptionShards(tmp_path / "index.json", tmp_path / "rowmap.npz", tmp_path / "ids.txt")
    emb, msk = idx.get(2)
    np.testing.assert_array_equal(emb, np.load(tmp_path / "emb_1.npy")[0])
    assert msk.shape == (5, 77)


@pytest.mark.research_data
def test_real_metadata_and_legacy_ocr(research_root):
    ds = research_root / "XSTAR_dataset"
    m = Manifest.load(ds / "XSTAR_final_manifest2.json")
    assert m.validate() == []
    for split, n in (("train", 4515), ("val", 645), ("test", 1289)):
        assert split_order(m, split, ds / "XSTAR_embeddings" / f"image_filenames_{split}.txt") == m.split_filenames(split)
        assert len(m.split_records(split)) == n
    f = ds / "XSTAR_embeddings" / "ocr_test_vitb32_lines" / "A_image_103.npz"
    E = load_ocr_embeddings(f)
    lines = read_ocr_lines(f, allow_legacy_tokens=True)
    assert E.shape == (len(lines), 512)
    assert lines == normalize_ocr_lines(m.record("A_image_103")["ocr_lines_en"])
