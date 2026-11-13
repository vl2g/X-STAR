#!/usr/bin/env python3
"""Read-only audit of the X-STAR research artifacts selected for release.

The utility inspects an existing local X-STAR layout (manifest, split lists,
cached CLIP features, OCR line embeddings and checkpoints) and writes a small
JSON report. It never modifies its inputs and never runs OCR, CLIP,
translation, training or inference.

Safety rules followed throughout:
  * .npy arrays are opened with ``allow_pickle=False`` (memory-mapped when large).
  * OCR ``tokens`` arrays are pickled object arrays; they are decoded with a
    restricted unpickler that only permits NumPy array reconstruction and plain
    Python strings.
  * Checkpoints are inspected with ``torch.load(..., weights_only=True)`` on CPU.
  * The report is refused if it would be written inside an input root.

Example:
    python tools/audit_release_inputs.py \\
        --dataset-root /path/to/XSTAR_dataset \\
        --textcaps-checkpoint /path/to/textcaps/best.pt
"""

import argparse
import hashlib
import io
import json
import pickle
import platform
import random
import sys
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "audit" / "release_inputs_audit.json"

SPLITS = ("train", "val", "test")
TRANSLATED_LANGS = ("assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam",
                    "marathi", "odia", "punjabi", "tamil", "telugu", "chinese")
QUERY_FIELDS = ("english_query",) + tuple(f"{lang}_query" for lang in TRANSLATED_LANGS)
TO_ENGLISH_FIELDS = tuple(f"{lang}_to_english" for lang in TRANSLATED_LANGS)
# Evaluation query sets: the English original plus each back-translation to English.
EVAL_TAGS = ("english_query",) + TO_ENGLISH_FIELDS
CHINESE_FOLDER = "N"

CLIP_DIM = 512
VISUAL_TOKENS = 50
TEXT_CTX = 77

MANIFEST_VARIANTS = {
    "primary": "XSTAR_final_manifest2.json",
    "gt_ocr": "XSTAR_final_manifest_with_gt_lines_en.json",
    "ocr_crops": "XSTAR_final_manifest_with_crops.json",
}
OCR_DIRS = {
    "recognized": "ocr_{split}_vitb32_lines",
    "gt": "gt_ocr_{split}_vitb32_lines",
    "corrected": "ocr_corrected_{split}_vitb32_lines",
}
CORRECT_GLOBAL_DIR = "to_english_vitb32_2"
MISSPELLED_GLOBAL_DIR = "to_english_vitb3_2"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sanitize(text):
    return " ".join((text or "").split())


def normalized_lines(lines):
    """Whitespace-collapsed non-empty lines, as used when encoding OCR lines."""
    return [sanitize(s) for s in (lines or []) if sanitize(s)]


def is_latin_letter(c):
    return c.isalpha() and unicodedata.name(c, "").startswith("LATIN")


def has_non_latin_letters(text):
    return any(c.isalpha() and not is_latin_letter(c) for c in text)


def image_id_from_filename(rel_path):
    p = Path(rel_path)
    return f"{p.parts[0]}_{p.stem}" if len(p.parts) > 1 else p.stem


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Report:
    def __init__(self, roots):
        self.roots = {k: str(v) for k, v in roots.items() if v is not None}
        self.checks = []
        self.checksums = []

    def rel(self, path):
        """Express a path relative to the most specific matching input root."""
        path = Path(path).resolve()
        best = None
        for name, root in self.roots.items():
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if best is None or len(str(rel)) < len(best[1]):
                best = (name, str(rel))
        return f"<{best[0]}>/{best[1]}" if best else str(path)

    def add(self, check_id, status, scope, details):
        assert status in {"pass", "fail", "warn", "info", "skipped"}
        assert scope in {"metadata", "full", "sampled", "presence", "n/a"}
        self.checks.append({"id": check_id, "status": status, "scope": scope, "details": details})
        print(f"[{status.upper():7}] {check_id}")

    def checksum(self, role, path):
        path = Path(path)
        if not path.is_file():
            self.checksums.append({"role": role, "path": self.rel(path), "exists": False})
            return
        self.checksums.append({"role": role, "path": self.rel(path), "exists": True,
                               "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})


def npy_header(fileobj):
    version = np.lib.format.read_magic(fileobj)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fileobj)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fileobj)
    return shape, fortran, dtype


class _RestrictedUnpickler(pickle.Unpickler):
    """Only allows what np.save emits for an object array of Python strings."""
    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_npz_member(zf, name):
    """Return (shape, dtype_str, array-or-None) for one member of an .npz archive."""
    with zf.open(name) as f:
        shape, fortran, dtype = npy_header(f)
        if dtype.hasobject:
            arr = _RestrictedUnpickler(io.BytesIO(f.read())).load()
            if not all(isinstance(x, str) for x in arr.ravel().tolist()):
                raise ValueError(f"{name}: object array contains non-string values")
            return shape, "object(str)", arr
        data = f.read()
    arr = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape)) if shape else 1)
    arr = arr.reshape(shape, order="F" if fortran else "C")
    return shape, str(dtype), arr


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_manifest(rep, manifest_path, dataset_root):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest["data"] if isinstance(manifest, dict) else manifest
    meta = manifest.get("meta", {}) if isinstance(manifest, dict) else {}

    required = ("image_id", "filename", "language", "split", "ocr_lines_en") + QUERY_FIELDS + TO_ENGLISH_FIELDS
    missing_fields = {k: sum(1 for r in records if k not in r) for k in required}
    missing_fields = {k: v for k, v in missing_fields.items() if v}
    empty_queries = {k: sum(1 for r in records if not sanitize(r.get(k))) for k in QUERY_FIELDS + TO_ENGLISH_FIELDS}
    empty_queries = {k: v for k, v in empty_queries.items() if v}

    ids = [r.get("image_id") for r in records]
    filenames = [r.get("filename") for r in records]
    id_mismatch = [r["image_id"] for r in records if r.get("image_id") != image_id_from_filename(r.get("filename", ""))]
    folder_mismatch = [r["image_id"] for r in records if r.get("language") != Path(r.get("filename", "")).parts[0]]
    split_counts = {s: sum(1 for r in records if r.get("split") == s) for s in SPLITS}
    other_split = sum(1 for r in records if r.get("split") not in SPLITS)
    folders = {}
    for r in records:
        folders[r.get("language")] = folders.get(r.get("language"), 0) + 1
    n_queries = sum(1 for r in records for k in QUERY_FIELDS if sanitize(r.get(k)))
    n_test_queries = sum(1 for r in records if r.get("split") == "test" for k in QUERY_FIELDS if sanitize(r.get(k)))
    empty_ocr = sum(1 for r in records if not r.get("ocr_lines_en"))
    non_latin_english = sorted(r["image_id"] for r in records if has_non_latin_letters(r.get("english_query", "")))
    absent_images = [r["filename"] for r in records if not (dataset_root / r["filename"]).is_file()]

    ok = (not missing_fields and not empty_queries and len(set(ids)) == len(ids)
          and len(set(filenames)) == len(filenames) and not id_mismatch and not folder_mismatch
          and other_split == 0)
    rep.add("manifest.schema_and_ids", "pass" if ok else "fail", "full", {
        "manifest": rep.rel(manifest_path),
        "records": len(records),
        "split_counts": split_counts,
        "records_with_unknown_split": other_split,
        "records_per_folder": dict(sorted(folders.items())),
        "unique_image_ids": len(set(ids)) == len(ids),
        "unique_filenames": len(set(filenames)) == len(filenames),
        "image_id_equals_folder_underscore_stem": not id_mismatch,
        "language_field_equals_folder": not folder_mismatch,
        "missing_required_fields": missing_fields,
        "empty_query_fields": empty_queries,
        "meta_keys": sorted(meta.keys()),
        "meta_exclude_labels": meta.get("exclude_labels"),
        "meta_split_ratios": meta.get("ratios"),
        "meta_seed": meta.get("seed"),
    })
    rep.add("manifest.query_counts", "pass" if n_queries == 13 * len(records) else "fail", "full", {
        "query_languages": len(QUERY_FIELDS),
        "query_fields": list(QUERY_FIELDS),
        "queries_total": n_queries,
        "queries_test": n_test_queries,
        "back_translated_fields": list(TO_ENGLISH_FIELDS),
    })
    without_latin = {r["image_id"]: {"split": r["split"], "english_query": r["english_query"]}
                     for r in records if not any(is_latin_letter(c) for c in r.get("english_query", ""))}
    split_of = {r["image_id"]: r["split"] for r in records}
    mixed_by_split = {s: sum(1 for i in non_latin_english if split_of[i] == s) for s in SPLITS}
    rep.add("manifest.content_flags", "warn" if without_latin else "info", "full", {
        "records_with_empty_ocr_lines_en": empty_ocr,
        "english_query_containing_non_latin_letters": len(non_latin_english),
        "english_query_containing_non_latin_letters_by_split": mixed_by_split,
        "english_query_containing_non_latin_letters_examples": non_latin_english[:10],
        "english_query_without_any_latin_letters": without_latin,
        "note": ("Empty OCR is expected for some images. Native-script scene text inside an English query can be "
                 "intentional; an english_query with no Latin letters at all needs manual review."),
    })
    rep.add("manifest.source_images_present", "pass" if not absent_images else "warn", "presence", {
        "images_checked": len(records),
        "absent": absent_images[:20],
        "absent_count": len(absent_images),
        "note": "File presence only under the dataset root; image contents were not hashed.",
    })
    return records


def check_manifest_variants(rep, dataset_root, primary_records):
    primary = {r["image_id"]: r for r in primary_records}
    for role, name in MANIFEST_VARIANTS.items():
        path = dataset_root / name
        rep.checksum(f"manifest.{role}", path)
        if role == "primary" or not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        extra = sorted({k for r in data for k in r} - {k for r in primary_records for k in r})
        same_ids = {r["image_id"] for r in data} == set(primary)
        field_diffs = []
        for r in data:
            p = primary.get(r["image_id"])
            if p is None:
                continue
            for k in QUERY_FIELDS + TO_ENGLISH_FIELDS + ("split", "filename", "ocr_lines_en"):
                if r.get(k) != p.get(k):
                    field_diffs.append({"image_id": r["image_id"], "field": k})
        details = {"manifest": rep.rel(path), "records": len(data), "same_image_ids_as_primary": same_ids,
                   "fields_not_in_primary": extra, "shared_field_differences": field_diffs[:20],
                   "shared_field_difference_count": len(field_diffs)}
        if role == "gt_ocr":
            src = {}
            for r in data:
                key = f"{'N' if r.get('language') == CHINESE_FOLDER else 'non-N'}:{r.get('gt_source')}"
                src[key] = src.get(key, 0) + 1
            details["gt_source_counts"] = src
            details["gt_lines_equal_primary_ocr_lines"] = sum(
                1 for r in data if r.get("gt_lines_en") == primary.get(r["image_id"], {}).get("ocr_lines_en"))
        rep.add(f"manifest_variant.{role}", "info", "full", details)


def check_split_lists(rep, emb_root, records):
    by_split = {s: {r["filename"] for r in records if r["split"] == s} for s in SPLITS}
    lists = {}
    for s in SPLITS:
        path = emb_root / f"image_filenames_{s}.txt"
        rep.checksum(f"split_list.{s}", path)
        lists[s] = read_lines(path)
    sets = {s: set(v) for s, v in lists.items()}
    pairwise = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    details = {
        "counts": {s: len(v) for s, v in lists.items()},
        "duplicates_within_list": {s: len(v) - len(sets[s]) for s, v in lists.items()},
        "pairwise_overlap": pairwise,
        "matches_manifest_split_membership": {s: sets[s] == by_split[s] for s in SPLITS},
        "union_equals_manifest": set().union(*sets.values()) == {r["filename"] for r in records},
    }
    ok = (all(v == 0 for v in pairwise.values()) and all(details["matches_manifest_split_membership"].values())
          and details["union_equals_manifest"] and not any(details["duplicates_within_list"].values()))
    rep.add("splits.disjoint_and_match_manifest", "pass" if ok else "fail", "full", details)
    return {s: [image_id_from_filename(fn) for fn in lists[s]] for s in SPLITS}


def check_image_features(rep, emb_root, split_ids, sample_rows, rng):
    details, ok = {}, True
    for s in SPLITS:
        path = emb_root / f"{s}_image_tokens_postxfmr_vitb32.npy"
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        rows = sorted(rng.sample(range(arr.shape[0]), min(sample_rows, arr.shape[0])))
        sample = np.asarray(arr[rows], dtype=np.float32)
        expected = (len(split_ids[s]), VISUAL_TOKENS, CLIP_DIM)
        good = tuple(arr.shape) == expected and arr.dtype == np.float32 and np.isfinite(sample).all()
        ok &= good
        details[s] = {"path": rep.rel(path), "shape": list(arr.shape), "expected_shape": list(expected),
                      "dtype": str(arr.dtype), "size_bytes": path.stat().st_size,
                      "sampled_rows_finite": bool(np.isfinite(sample).all()), "sampled_rows": len(rows),
                      "sample_token_norm_mean": float(np.linalg.norm(sample, axis=-1).mean())}
    gpath = emb_root / "test_image_global_embeddings.npy"
    g = np.load(gpath, allow_pickle=False)
    gnorm = np.linalg.norm(g, axis=-1)
    good = tuple(g.shape) == (len(split_ids["test"]), CLIP_DIM) and np.isfinite(g).all()
    ok &= good
    details["test_global"] = {"path": rep.rel(gpath), "shape": list(g.shape), "dtype": str(g.dtype),
                              "all_finite": bool(np.isfinite(g).all()),
                              "l2_norm_min": float(gnorm.min()), "l2_norm_max": float(gnorm.max())}
    details["note"] = ("Token arrays: shape/dtype from memory-mapped headers plus finiteness on sampled rows. "
                       "Row order is assumed to follow image_filenames_<split>.txt; the arrays carry no IDs.")
    rep.add("image_features.shapes", "pass" if ok else "fail", "sampled", details)


def check_packed_queries(rep, emb_root, split_ids):
    packed = emb_root / "colbert_packed2"
    details, ok = {}, True
    sets = [(s, "", f"{s}_q") for s in SPLITS] + [("test", tag, "test_q") for tag in TO_ENGLISH_FIELDS]
    for split, tag, stem in sets:
        suffix = f"__{tag}" if tag else ""
        name = f"{split}:{tag or 'english_query'}"
        ids_path = packed / f"{stem}_ids{suffix}.txt"
        tok = np.load(packed / f"{stem}_tok{suffix}.npy", mmap_mode="r", allow_pickle=False)
        msk = np.load(packed / f"{stem}_mask{suffix}.npy", mmap_mode="r", allow_pickle=False)
        ids = read_lines(ids_path)
        n = len(split_ids[split])
        good = (ids == split_ids[split] and tuple(tok.shape) == (n, 1, TEXT_CTX, CLIP_DIM)
                and tuple(msk.shape) == (n, 1, TEXT_CTX) and tok.dtype == np.float32 and msk.dtype == np.bool_)
        ok &= good
        details[name] = {"ids_match_image_list_order": ids == split_ids[split],
                         "tok_shape": list(tok.shape), "tok_dtype": str(tok.dtype),
                         "mask_shape": list(msk.shape), "mask_dtype": str(msk.dtype)}
        rep.checksum(f"packed_query_ids.{name}", ids_path)
    rep.add("packed_queries.alignment_and_shapes", "pass" if ok else "fail", "metadata", details)


def check_global_queries(rep, emb_root, split_ids, records):
    by_id = {r["image_id"]: r for r in records}
    packed = emb_root / "colbert_packed2"
    details, ok = {}, True
    for tag in EVAL_TAGS:
        if tag == "english_query":
            d = emb_root / "english_query_vitb32_2"
            vec, idp, strp = d / "test_english_query_vitb32.npy", d / "test_image_ids.txt", d / "test_english_query_strings.txt"
            packed_ids = packed / "test_q_ids.txt"
        else:
            d = emb_root / CORRECT_GLOBAL_DIR
            vec, idp, strp = d / f"test_{tag}_vitb32.npy", d / f"test_{tag}_image_ids.txt", d / f"test_{tag}_strings.txt"
            packed_ids = packed / f"test_q_ids__{tag}.txt"
        missing = [p.name for p in (vec, idp, strp) if not p.is_file()]
        if missing:
            ok = False
            details[tag] = {"missing": missing}
            continue
        arr = np.load(vec, allow_pickle=False)
        ids, strings = read_lines(idp), read_lines(strp)
        expected = [sanitize(by_id[i][tag]) for i in ids]
        mismatched = [ids[i] for i, (a, b) in enumerate(zip(strings, expected)) if a != b]
        norms = np.linalg.norm(arr, axis=-1)
        good = (tuple(arr.shape) == (len(ids), CLIP_DIM) and np.isfinite(arr).all()
                and ids == read_lines(packed_ids) and set(ids) == set(split_ids["test"])
                and len(ids) == len(split_ids["test"]) and len(strings) == len(ids) and not mismatched)
        ok &= good
        details[tag] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "all_finite": bool(np.isfinite(arr).all()),
                        "l2_norm_min": float(norms.min()), "l2_norm_max": float(norms.max()),
                        "ids_equal_packed_ids": ids == read_lines(packed_ids),
                        "covers_full_test_gallery": set(ids) == set(split_ids["test"]),
                        "cached_strings_match_manifest": len(ids) - len(mismatched),
                        "cached_strings_mismatch": mismatched[:10], "rows": len(ids)}
        rep.checksum(f"global_query_ids.{tag}", idp)
    total = sum(v.get("cached_strings_match_manifest", 0) for v in details.values())
    rep.add("baseline_global_queries.all_13_sets", "pass" if ok else "fail", "full", {
        "directory_used": CORRECT_GLOBAL_DIR,
        "correct_directory_exists": (emb_root / CORRECT_GLOBAL_DIR).is_dir(),
        "misspelled_directory_exists": (emb_root / MISSPELLED_GLOBAL_DIR).exists(),
        "query_sets_present": sum(1 for v in details.values() if "missing" not in v),
        "cached_test_strings_matching_manifest_total": total,
        "per_set": details,
        "note": "String comparison uses whitespace-collapsed manifest fields, matching the original extractor.",
    })


def check_ocr(rep, emb_root, split_ids, records, mode, sample_size, rng):
    lines_by_id = {r["image_id"]: normalized_lines(r.get("ocr_lines_en")) for r in records}
    summary, ok = {}, True
    for s in SPLITS:
        d = emb_root / OCR_DIRS["recognized"].format(split=s)
        expected = split_ids[s]
        present = {p.stem for p in d.glob("*.npz")}
        missing = [i for i in expected if i not in present]
        missing_set = set(missing)
        extra = sorted(present - set(expected))
        if mode == "full":
            inspect = expected
        elif mode == "sample":
            inspect = rng.sample(expected, min(sample_size, len(expected)))
        else:
            inspect = []
        stats = {"inspected": 0, "bad_schema": [], "non_finite": [], "token_count_mismatch": [],
                 "tokens_differ_from_manifest": [], "lines_total": 0, "lines_max": 0,
                 "images_with_more_than_40_lines": 0, "images_with_zero_lines": 0,
                 "norm_min": float("inf"), "norm_max": 0.0}
        for iid in inspect:
            if iid in missing_set:
                continue
            with zipfile.ZipFile(d / f"{iid}.npz") as zf:
                names = set(zf.namelist())
                if not {"E.npy", "tokens.npy"} <= names:
                    stats["bad_schema"].append(iid)
                    continue
                shape, dtype, E = load_npz_member(zf, "E.npy")
                _, _, tokens = load_npz_member(zf, "tokens.npy")
                scores_shape = load_npz_member(zf, "scores.npy")[0] if "scores.npy" in names else None
            stats["inspected"] += 1
            if len(shape) != 2 or shape[1] != CLIP_DIM or dtype != "float32":
                stats["bad_schema"].append(iid)
                continue
            if not np.isfinite(E).all():
                stats["non_finite"].append(iid)
            if len(tokens) != shape[0] or (scores_shape is not None and scores_shape != (shape[0],)):
                stats["token_count_mismatch"].append(iid)
            if [sanitize(t) for t in tokens.tolist()] != lines_by_id[iid]:
                stats["tokens_differ_from_manifest"].append(iid)
            stats["lines_total"] += shape[0]
            stats["lines_max"] = max(stats["lines_max"], shape[0])
            stats["images_with_more_than_40_lines"] += shape[0] > 40
            stats["images_with_zero_lines"] += shape[0] == 0
            if shape[0]:
                n = np.linalg.norm(E, axis=-1)
                stats["norm_min"] = min(stats["norm_min"], float(n.min()))
                stats["norm_max"] = max(stats["norm_max"], float(n.max()))
        for key in ("bad_schema", "non_finite", "token_count_mismatch", "tokens_differ_from_manifest"):
            stats[f"{key}_count"] = len(stats[key])
            stats[key] = stats[key][:10]
        if stats["norm_min"] == float("inf"):
            stats["norm_min"] = None
        good = not missing and not any(stats[f"{k}_count"] for k in
                                       ("bad_schema", "non_finite", "token_count_mismatch"))
        ok &= good
        summary[s] = {"directory": rep.rel(d), "expected": len(expected), "present": len(present),
                      "missing": missing[:10], "missing_count": len(missing), "unexpected_extra": extra[:10],
                      "unexpected_extra_count": len(extra), **stats}
    variants = {}
    for kind in ("gt", "corrected"):
        variants[kind] = {}
        for s in SPLITS:
            d = emb_root / OCR_DIRS[kind].format(split=s)
            present = {p.stem for p in d.glob("*.npz")} if d.is_dir() else set()
            variants[kind][s] = {"directory": rep.rel(d), "exists": d.is_dir(),
                                 "expected_ids_present": len(set(split_ids[s]) & present),
                                 "expected": len(split_ids[s])}
    scope = {"full": "full", "sample": "sampled", "presence": "presence"}[mode]
    rep.add("ocr.recognized_line_embeddings", "pass" if ok else "fail", scope, {
        "inspection_mode": mode,
        "per_split": summary,
        "what_was_checked": ("presence for every expected image; for inspected files: member schema, "
                             "E shape [lines,512] float32, finiteness, L2 norms, token/score lengths, "
                             "token strings vs manifest ocr_lines_en"),
        "not_checked": "Embedding values were not recomputed with CLIP.",
    })
    rep.add("ocr.variant_directories", "info", "presence", {
        "note": "GT-OCR and corrected-OCR caches are separate experiment variants, not primary inputs.",
        **variants})


def check_translated_ocr(rep, path, records):
    with open(path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    flat = {fn: v.get("translated_lines") for folder in merged.values() for fn, v in folder.items()}
    non_n = [r for r in records if r["language"] != CHINESE_FOLDER]
    found = [r for r in non_n if r["filename"] in flat]
    exact = {r["image_id"] for r in found if flat[r["filename"]] == r["ocr_lines_en"]}
    normalized = {r["image_id"] for r in found
                  if normalized_lines(flat[r["filename"]]) == normalized_lines(r["ocr_lines_en"])}
    ok = len(found) == len(normalized) == len(non_n) == len(flat)
    rep.add("ocr.translated_artifact_matches_manifest", "pass" if ok else "fail", "full", {
        "artifact": rep.rel(path), "artifact_records": len(flat), "manifest_non_chinese_records": len(non_n),
        "found_by_filename": len(found),
        "raw_list_equal": len(exact),
        "equal_after_line_normalization": len(normalized),
        "differ_only_in_whitespace_or_empty_lines": sorted(normalized - exact)[:10],
        "differ_only_in_whitespace_or_empty_lines_count": len(normalized - exact),
        "mismatches_after_normalization": [r["image_id"] for r in found if r["image_id"] not in normalized][:10],
        "normalization": "collapse internal whitespace, strip, drop empty lines (as the OCR line encoder does)",
    })


def check_chinese_mapping(rep, mapping_path, records, dataset_root, rctw_root, verify_hashes):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)["mapping"]
    n_files = {r["filename"] for r in records if r["language"] == CHINESE_FOLDER}
    mapped = {m["xstar_filename"] for m in mapping}
    same_basename = all(Path(m["xstar_filename"]).name == Path(m["source_relative_path"]).name for m in mapping)
    details = {"mapping": rep.rel(mapping_path), "entries": len(mapping),
               "manifest_chinese_records": len(n_files), "covers_exactly_manifest_chinese_records": mapped == n_files,
               "source_directories": sorted({str(Path(m["source_relative_path"]).parent) for m in mapping}),
               "basename_preserved": same_basename}
    ok = mapped == n_files and len(mapping) == len(mapped)
    scope = "metadata"
    if verify_hashes and rctw_root is not None:
        scope = "full"
        bad = []
        for m in mapping:
            src = rctw_root / m["source_relative_path"]
            dst = dataset_root / m["xstar_filename"]
            if not (src.is_file() and dst.is_file() and sha256_file(src) == m["sha256"] == sha256_file(dst)):
                bad.append(m["xstar_filename"])
        details["sha256_verified_both_sides"] = len(mapping) - len(bad)
        details["sha256_failures"] = bad[:10]
        ok &= not bad
    rep.add("chinese_subset.rctw17_mapping", "pass" if ok else "fail", scope, details)


def check_checkpoint(rep, role, path):
    rep.checksum(f"checkpoint.{role}", path)
    try:
        import torch
    except ImportError:
        rep.add(f"checkpoint.{role}.metadata", "skipped", "n/a", {"reason": "PyTorch not importable"})
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", {})
    blocks = sorted({int(k.split(".")[1]) for k in state if k.startswith("blocks.")})
    tau = state.get("tau")
    null_o = state.get("null_O")
    rep.add(f"checkpoint.{role}.metadata", "info", "metadata", {
        "path": rep.rel(path),
        "loaded_with": "torch.load(weights_only=True, map_location='cpu')",
        "top_level_keys": sorted(ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "config": ckpt.get("config"),
        "metrics": ckpt.get("metrics"),
        "model_tensors": len(state),
        "model_parameters": int(sum(v.numel() for v in state.values())),
        "fusion_blocks": len(blocks),
        "stored_tau": None if tau is None else float(tau),
        "null_O_shape": None if null_o is None else list(null_o.shape),
        "has_optimizer_state": "optim" in ckpt,
    })


def check_existing_eval_json(rep, emb_root):
    path = emb_root / "colbert_vs_clip_eval_test_XSTAR.json"
    if not path.is_file():
        rep.add("legacy_eval_json.identity", "skipped", "n/a", {"reason": "file not present"})
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    modes = {tag: v.get("clip_mode") for tag, v in payload.get("results", {}).items()}
    ckpt = str(payload.get("checkpoint", ""))
    rep.add("legacy_eval_json.identity", "warn", "metadata", {
        "path": rep.rel(path),
        "checkpoint_field": ckpt.split("XSTAR_embeddings/")[-1],
        "references_main_checkpoint": ckpt.endswith("checkpoints_colbert_XSTAR_5/best.pt"),
        "clip_text_modes": sorted({m.get("text") for m in modes.values() if m}),
        "note": "Not the authoritative main-result file. Do not overwrite or reuse it as the main result.",
    })


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=Path, required=True,
                    help="X-STAR dataset root containing the manifest(s) and image folders.")
    ap.add_argument("--embeddings-root", type=Path,
                    help="Feature cache root (default: <dataset-root>/XSTAR_embeddings).")
    ap.add_argument("--manifest", default=MANIFEST_VARIANTS["primary"],
                    help="Primary manifest filename inside the dataset root.")
    ap.add_argument("--main-checkpoint", type=Path,
                    help="Main X-STAR checkpoint (default: <embeddings-root>/checkpoints_colbert_XSTAR_5/best.pt).")
    ap.add_argument("--textcaps-checkpoint", type=Path, help="TextCaps initialization checkpoint.")
    ap.add_argument("--translated-ocr", type=Path, help="Recovered merged translated-OCR JSON to compare.")
    ap.add_argument("--chinese-mapping", type=Path, help="Chinese subset to RCTW-17 mapping JSON.")
    ap.add_argument("--rctw-root", type=Path, help="RCTW-17 root that the mapping's source paths are relative to.")
    ap.add_argument("--verify-chinese-hashes", action="store_true",
                    help="Hash the mapped Chinese images on both sides (500 + 500 files).")
    ap.add_argument("--ocr-mode", choices=("full", "sample", "presence"), default="full")
    ap.add_argument("--ocr-sample-size", type=int, default=200)
    ap.add_argument("--feature-sample-rows", type=int, default=64)
    ap.add_argument("--skip-checkpoints", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    emb_root = (args.embeddings_root or dataset_root / "XSTAR_embeddings").resolve()
    main_ckpt = args.main_checkpoint or emb_root / "checkpoints_colbert_XSTAR_5" / "best.pt"
    output = args.output.resolve()
    roots = {"dataset_root": dataset_root, "embeddings_root": emb_root,
             "textcaps_checkpoint_dir": args.textcaps_checkpoint.resolve().parent if args.textcaps_checkpoint else None,
             "audit_inputs": args.translated_ocr.resolve().parent if args.translated_ocr else None,
             "rctw_root": args.rctw_root.resolve() if args.rctw_root else None}
    for root in filter(None, roots.values()):
        if output == root or root in output.parents:
            sys.exit(f"Refusing to write the report inside an input root: {output}")

    rng = random.Random(args.seed)
    rep = Report(roots)
    records = check_manifest(rep, dataset_root / args.manifest, dataset_root)
    check_manifest_variants(rep, dataset_root, records)
    split_ids = check_split_lists(rep, emb_root, records)
    check_image_features(rep, emb_root, split_ids, args.feature_sample_rows, rng)
    check_packed_queries(rep, emb_root, split_ids)
    check_global_queries(rep, emb_root, split_ids, records)
    check_ocr(rep, emb_root, split_ids, records, args.ocr_mode, args.ocr_sample_size, rng)
    if args.translated_ocr:
        check_translated_ocr(rep, args.translated_ocr, records)
    if args.chinese_mapping:
        check_chinese_mapping(rep, args.chinese_mapping, records, dataset_root, args.rctw_root,
                              args.verify_chinese_hashes)
    check_existing_eval_json(rep, emb_root)
    if not args.skip_checkpoints:
        check_checkpoint(rep, "xstar_main", main_ckpt)
        if args.textcaps_checkpoint:
            check_checkpoint(rep, "textcaps_init", args.textcaps_checkpoint)

    status_counts = {}
    for c in rep.checks:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        pass
    payload = {
        "tool": "tools/audit_release_inputs.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "read_only": True,
        "versions": versions,
        "roots": rep.roots,
        "options": {"ocr_mode": args.ocr_mode, "ocr_sample_size": args.ocr_sample_size,
                    "feature_sample_rows": args.feature_sample_rows, "seed": args.seed,
                    "verify_chinese_hashes": args.verify_chinese_hashes},
        "status_counts": status_counts,
        "checks": rep.checks,
        "checksums": rep.checksums,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Status counts: {status_counts}")
    print(f"Report: {output}")
    return 1 if status_counts.get("fail") else 0


if __name__ == "__main__":
    sys.exit(main())
