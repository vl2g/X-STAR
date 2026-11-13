"""Manifest, split lists, cached features and training datasets.

Conventions recovered from the research code:

* ``filename`` is ``<folder>/<name>.jpg``; ``image_id`` is ``<folder>_<stem>``.
* Split image lists are ordered; every per-split feature array follows that
  order. For X-STAR the lists equal the sorted filenames of each split.
* Text is normalised by collapsing whitespace (``" ".join(s.split())``) before
  CLIP tokenisation, for queries and OCR lines alike; empty OCR lines are dropped.

Every loader checks shapes and ID alignment and raises ``DataError`` instead of
skipping, padding or substituting data.
"""

import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np

from xstar.constants import CLIP_DIM, NATIVE_QUERY_FIELDS, OCR_FORMAT, QUERY_SETS, SPLITS, TEXT_CONTEXT, VISUAL_TOKENS
from xstar.errors import DataError

# ---------------------------------------------------------------------------
# Text and identifiers
# ---------------------------------------------------------------------------

def sanitize(text):
    """Collapse whitespace, as the original extractors did before tokenisation."""
    return " ".join((text or "").split())


def normalize_ocr_lines(lines):
    """Whitespace-collapsed, non-empty OCR lines in their original order."""
    return [s for s in (sanitize(x) for x in (lines or [])) if s]


def image_id_from_filename(rel_path):
    p = Path(rel_path)
    return f"{p.parts[0]}_{p.stem}" if len(p.parts) > 1 else p.stem


def read_lines(path):
    """Non-empty stripped lines (the research ``load_list``)."""
    if not Path(path).is_file():
        raise DataError(f"missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class Manifest:
    """The X-STAR manifest: ``{"meta": {...}, "data": [record, ...]}``."""

    def __init__(self, records, meta=None, path=None):
        self.records = records
        self.meta = meta or {}
        self.path = path
        self.by_id = {}
        self.by_filename = {}
        for r in records:
            self.by_id.setdefault(r.get("image_id"), r)
            self.by_filename.setdefault(r.get("filename"), r)

    @classmethod
    def load(cls, path):
        if not Path(path).is_file():
            raise DataError(f"manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return cls(obj["data"], obj.get("meta"), path)
        if isinstance(obj, list):
            return cls(obj, None, path)
        raise DataError(f"{path}: expected a list of records or a mapping with a 'data' list")

    def split_records(self, split):
        return [r for r in self.records if r.get("split") == split]

    def split_filenames(self, split):
        """Canonical split order: sorted filenames (matches the released split lists)."""
        return sorted(r["filename"] for r in self.split_records(split))

    def record(self, image_id):
        if image_id not in self.by_id:
            raise DataError(f"image ID {image_id!r} not in manifest")
        return self.by_id[image_id]

    def query_text(self, image_id, field):
        """Normalised text of a query field for one image; raises if empty."""
        text = sanitize(self.record(image_id).get(field))
        if not text:
            raise DataError(f"empty {field!r} for {image_id}")
        return text

    def validate(self, required_fields=("image_id", "filename", "split") + NATIVE_QUERY_FIELDS + QUERY_SETS[1:],
                 ocr_field="ocr_lines_en"):
        """Return a list of human-readable problems (empty when valid)."""
        problems = []
        fields = tuple(required_fields) + ((ocr_field,) if ocr_field else ())
        for k in fields:
            n = sum(1 for r in self.records if k not in r)
            if n:
                problems.append(f"{n} records lack field {k!r}")
        for k in tuple(f for f in fields if f.endswith("_query") or f.endswith("_to_english")):
            n = sum(1 for r in self.records if k in r and not sanitize(r[k]))
            if n:
                problems.append(f"{n} records have empty {k!r}")
        ids = [r.get("image_id") for r in self.records]
        fns = [r.get("filename") for r in self.records]
        if len(set(ids)) != len(ids):
            problems.append("duplicate image_id values")
        if len(set(fns)) != len(fns):
            problems.append("duplicate filename values")
        bad_id = [r.get("image_id") for r in self.records if r.get("image_id") != image_id_from_filename(r.get("filename") or "")]
        if bad_id:
            problems.append(f"{len(bad_id)} image_id values differ from <folder>_<stem>, e.g. {bad_id[:3]}")
        bad_path = [fn for fn in fns if not fn or os.path.isabs(fn) or ".." in Path(fn).parts or len(Path(fn).parts) != 2]
        if bad_path:
            problems.append(f"{len(bad_path)} filenames are not relative '<folder>/<file>' paths, e.g. {bad_path[:3]}")
        bad_split = [r.get("image_id") for r in self.records if r.get("split") not in SPLITS]
        if bad_split:
            problems.append(f"{len(bad_split)} records have a split outside {SPLITS}")
        if ocr_field:
            bad_ocr = [r.get("image_id") for r in self.records
                       if ocr_field in r and not isinstance(r[ocr_field], list)]
            if bad_ocr:
                problems.append(f"{len(bad_ocr)} records have non-list {ocr_field!r}")
        return problems


# ---------------------------------------------------------------------------
# Cached query features
# ---------------------------------------------------------------------------

class PackedQueries:
    """Per-token query embeddings ``[N, C, T, D]``, masks ``[N, C, T]`` and IDs.

    X-STAR files have C = 1. Arrays are memory-mapped.
    """

    def __init__(self, tok_path, mask_path, ids_path):
        for p in (tok_path, mask_path, ids_path):
            if not Path(p).is_file():
                raise DataError(f"missing packed-query file: {p}")
        self.tok = np.load(tok_path, mmap_mode="r", allow_pickle=False)
        self.mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        self.ids = read_lines(ids_path)
        self.paths = (Path(tok_path), Path(mask_path), Path(ids_path))
        if self.tok.ndim != 4 or self.mask.ndim != 3:
            raise DataError(f"packed queries must be [N,C,T,D] and [N,C,T]; got {self.tok.shape}, {self.mask.shape}")
        if not (self.tok.shape[0] == self.mask.shape[0] == len(self.ids)) or self.tok.shape[1:3] != self.mask.shape[1:3]:
            raise DataError(f"packed queries misaligned: tok{self.tok.shape}, mask{self.mask.shape}, ids {len(self.ids)}")
        if self.tok.shape[2] != TEXT_CONTEXT or self.tok.shape[3] != CLIP_DIM:
            raise DataError(f"unexpected packed token shape {self.tok.shape}")
        if self.mask.dtype != np.bool_:
            raise DataError(f"packed mask must be bool, got {self.mask.dtype}")

    def __len__(self):
        return len(self.ids)

    def single_caption(self, rows=None):
        """``[N, T, D]`` float32 tokens and ``[N, T]`` masks for C = 1 files."""
        if self.tok.shape[1] != 1:
            raise DataError(f"expected one query per row, file has C={self.tok.shape[1]}")
        sel = slice(None) if rows is None else rows
        return (np.array(self.tok[sel, 0], dtype=np.float32), np.array(self.mask[sel, 0], dtype=bool))


def load_global_queries(vec_path, ids_path):
    """Global CLIP query embeddings ``[N, D]`` and their IDs; both files required."""
    missing = [str(p) for p in (vec_path, ids_path) if not Path(p).is_file()]
    if missing:
        raise DataError("CLIP baseline needs global query embeddings and their ID file; missing: "
                        + ", ".join(missing) + ". Token averages are never used as a substitute.")
    arr = np.load(vec_path, allow_pickle=False)
    ids = read_lines(ids_path)
    if arr.ndim != 2 or arr.shape != (len(ids), CLIP_DIM):
        raise DataError(f"{vec_path}: shape {arr.shape} does not match {len(ids)} IDs x {CLIP_DIM}")
    if not np.isfinite(arr).all():
        raise DataError(f"{vec_path}: non-finite values")
    return arr.astype(np.float32, copy=False), ids


def load_image_global(path, expected_rows):
    if not Path(path).is_file():
        raise DataError(f"CLIP baseline needs global image embeddings; missing: {path}. "
                        "Patch-token averages are never used as a substitute.")
    arr = np.load(path, allow_pickle=False)
    if arr.shape != (expected_rows, CLIP_DIM):
        raise DataError(f"{path}: shape {arr.shape}, expected ({expected_rows}, {CLIP_DIM})")
    if not np.isfinite(arr).all():
        raise DataError(f"{path}: non-finite values")
    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Image tokens
# ---------------------------------------------------------------------------

class ImageTokens:
    """Visual tokens ``[N, 50, 512]`` paired with their ordered image list."""

    def __init__(self, tokens_path, image_list_path, mmap=True, id_fn=image_id_from_filename):
        if not Path(tokens_path).is_file():
            raise DataError(f"missing image tokens: {tokens_path}")
        self.path = Path(tokens_path)
        self.tokens = np.load(tokens_path, mmap_mode="r" if mmap else None, allow_pickle=False)
        self.filenames = read_lines(image_list_path)
        self.ids = [id_fn(fn) for fn in self.filenames]
        if self.tokens.ndim != 3 or self.tokens.shape[1:] != (VISUAL_TOKENS, CLIP_DIM):
            raise DataError(f"{tokens_path}: expected [N, {VISUAL_TOKENS}, {CLIP_DIM}], got {self.tokens.shape}")
        if self.tokens.shape[0] != len(self.ids):
            raise DataError(f"{tokens_path}: {self.tokens.shape[0]} rows but {len(self.ids)} images in {image_list_path}")
        if len(set(self.ids)) != len(self.ids):
            raise DataError(f"{image_list_path}: duplicate image IDs")
        self.row_of = {iid: i for i, iid in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def rows_for(self, ids):
        missing = [i for i in ids if i not in self.row_of]
        if missing:
            raise DataError(f"{len(missing)} IDs have no image tokens, e.g. {missing[:5]}")
        return np.array([self.row_of[i] for i in ids], dtype=np.int64)

    def get(self, row):
        arr = np.array(self.tokens[row], dtype=np.float32)
        if not np.isfinite(arr).all():
            raise DataError(f"non-finite image tokens at row {row} of {self.path}")
        return arr


# ---------------------------------------------------------------------------
# OCR line embeddings
# ---------------------------------------------------------------------------

def ocr_path(ocr_dir, image_id):
    return Path(ocr_dir) / f"{image_id}.npz"


def load_ocr_embeddings(path, dim=CLIP_DIM):
    """Numeric OCR line embeddings ``E`` ``[lines, dim]`` from a legacy or new file.

    Only the ``E`` member is read, with ``allow_pickle=False``; pickled token
    metadata in legacy files is never unpickled here.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"missing OCR file: {path} (an image without OCR lines must have an explicit empty file)")
    with np.load(path, allow_pickle=False) as z:
        if "E" not in z.files:
            raise DataError(f"{path}: no 'E' array")
        arr = z["E"]
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise DataError(f"{path}: E has shape {arr.shape}, expected (*, {dim})")
    if not np.isfinite(arr).all():
        raise DataError(f"{path}: non-finite OCR embeddings")
    return arr.astype(np.float32, copy=False)


class _StringArrayUnpickler(pickle.Unpickler):
    """Allows only what ``np.save`` emits for an object array of Python strings."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def read_ocr_lines(path, allow_legacy_tokens=False):
    """OCR line strings stored next to ``E``.

    New files store ``lines`` as a Unicode array (no pickle). Legacy files store
    ``tokens`` as a pickled object array; reading them requires
    ``allow_legacy_tokens=True`` and uses a restricted unpickler that accepts
    only NumPy array reconstruction and plain strings.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if "lines" in z.files:
            return [str(x) for x in z["lines"].tolist()]
    if not allow_legacy_tokens:
        raise DataError(f"{path}: legacy file with pickled 'tokens'; pass allow_legacy_tokens=True to read it safely")
    with zipfile.ZipFile(path) as zf, zf.open("tokens.npy") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        _, _, dtype = reader(f)
        if not dtype.hasobject:
            raise DataError(f"{path}: unexpected non-object tokens array")
        arr = _StringArrayUnpickler(io.BytesIO(f.read())).load()
    values = arr.tolist()
    if not all(isinstance(x, str) for x in values):
        raise DataError(f"{path}: tokens contain non-string values")
    return values


def write_ocr_file(path, embeddings, lines):
    """Write one image's OCR lines in the pickle-free ``xstar-ocr-lines-v1`` format.

    Members: ``E`` float32 ``[L, 512]`` (L2-normalised CLIP text embedding per
    line), ``lines`` Unicode ``[L]`` (the encoded strings, in order), ``format``
    0-d Unicode array. Refuses to overwrite an existing file.
    """
    path = Path(path)
    E = np.asarray(embeddings, dtype=np.float32)
    if E.ndim != 2 or E.shape[1] != CLIP_DIM or E.shape[0] != len(lines):
        raise DataError(f"OCR embeddings {E.shape} do not match {len(lines)} lines")
    line_arr = np.array(list(lines), dtype=np.str_) if lines else np.zeros((0,), dtype="<U1")
    with open(path, "xb") as f:
        np.savez_compressed(f, E=E, lines=line_arr, format=np.array(OCR_FORMAT))


def pad_or_truncate(arr, K, D=CLIP_DIM):
    """First-K OCR lines and a validity mask (True = real line)."""
    T = arr.shape[0]
    if T >= K:
        return arr[:K], np.ones(K, dtype=np.bool_)
    out = np.zeros((K, D), dtype=np.float32)
    out[:T] = arr
    mask = np.zeros(K, dtype=np.bool_)
    mask[:T] = True
    return out, mask


class OcrBank:
    """Loads and pads OCR line embeddings for image IDs from one variant directory."""

    def __init__(self, ocr_dir, K):
        self.dir = Path(ocr_dir)
        if not self.dir.is_dir():
            raise DataError(f"OCR directory not found: {self.dir}")
        self.K = int(K)

    def get(self, image_id):
        return pad_or_truncate(load_ocr_embeddings(ocr_path(self.dir, image_id)), self.K)

    def check_ids(self, ids):
        missing = [i for i in ids if not ocr_path(self.dir, i).is_file()]
        if missing:
            raise DataError(f"{len(missing)} images have no OCR file in {self.dir}, e.g. {missing[:5]}")


# ---------------------------------------------------------------------------
# Training datasets (torch is imported lazily)
# ---------------------------------------------------------------------------

def _torch():
    import torch
    return torch


class PairDataset:
    """One (image, OCR, query captions) example per image.

    ``queries`` supplies ``get(i) -> (tok[C,T,D], mask[C,T])`` and ``ids``; each
    query row is matched to its image row by ID. Unmatched IDs raise.
    """

    def __init__(self, images, ocr_bank, queries, name):
        self.images, self.ocr, self.queries, self.name = images, ocr_bank, queries, name
        self.rows = images.rows_for(queries.ids)
        ocr_bank.check_ids(queries.ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        torch = _torch()
        image_id = self.queries.ids[i]
        V = self.images.get(int(self.rows[i]))
        O, mO = self.ocr.get(image_id)
        Q, M = self.queries.get(i)
        return {"V": torch.from_numpy(V), "O": torch.from_numpy(np.ascontiguousarray(O)),
                "mO": torch.from_numpy(mO), "Q": torch.from_numpy(np.asarray(Q, dtype=np.float32)),
                "Qm": torch.from_numpy(np.asarray(M, dtype=bool))}


def collate_pairs(batch):
    torch = _torch()
    return {k: torch.stack([b[k] for b in batch]) for k in ("V", "O", "mO", "Q", "Qm")}


class PackedQueryRows:
    """``get(i)`` adapter over :class:`PackedQueries` (X-STAR: C = 1)."""

    def __init__(self, packed):
        self.packed = packed
        self.ids = packed.ids

    def get(self, i):
        return np.array(self.packed.tok[i], dtype=np.float32), np.array(self.packed.mask[i], dtype=bool)


class TextCapsCaptionShards:
    """Sharded TextCaps caption tokens: 5 captions per image, recovered source format.

    ``index_json`` lists shards (``emb``/``mask`` .npy paths, rows), ``rowmap``
    (.npz with ``row2shard_ord`` and ``row2pos``) locates each global row and
    ``ids`` lists one image ID per row. Shard paths recorded on another machine
    are resolved by file name inside ``shard_dir`` (default: the index's folder).
    """

    def __init__(self, index_json, rowmap, ids, shard_dir=None, cache_size=12):
        meta = json.loads(Path(index_json).read_text(encoding="utf-8"))
        self.ids = read_lines(ids)
        if len(self.ids) != int(meta["total_rows"]):
            raise DataError(f"{ids}: {len(self.ids)} IDs but index lists {meta['total_rows']} rows")
        with np.load(rowmap, allow_pickle=False) as rm:
            self.row2shard = rm["row2shard_ord"].astype(np.int64)
            self.row2pos = rm["row2pos"].astype(np.int64)
        if len(self.row2shard) != len(self.ids):
            raise DataError(f"{rowmap}: row map length differs from ID count")
        base = Path(shard_dir) if shard_dir else Path(index_json).parent
        self.shards = []
        for s in meta["shards"]:
            pair = {}
            for kind in ("emb", "mask"):
                p = Path(s[kind])
                if not p.is_file():
                    p = base / p.name
                if not p.is_file():
                    raise DataError(f"caption shard not found: {s[kind]} (also looked in {base})")
                pair[kind] = p
            self.shards.append(pair)
        self._cache = {}
        self._cache_size = cache_size

    def _mmap(self, ordinal, kind):
        key = (ordinal, kind)
        if key not in self._cache:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = np.load(self.shards[ordinal][kind], mmap_mode="r", allow_pickle=False)
        return self._cache[key]

    def get(self, i):
        o, pos = int(self.row2shard[i]), int(self.row2pos[i])
        emb = np.array(self._mmap(o, "emb")[pos], dtype=np.float32)
        msk = np.array(self._mmap(o, "mask")[pos], dtype=bool)
        if emb.shape != (5, TEXT_CONTEXT, CLIP_DIM) or msk.shape != (5, TEXT_CONTEXT):
            raise DataError(f"caption row {i}: unexpected shapes {emb.shape}, {msk.shape}")
        return emb, msk


def split_order(manifest, split, image_list=None):
    """Ordered filenames for a split.

    Without ``image_list`` this is the canonical order (sorted filenames). A
    supplied list must contain exactly the manifest's images for that split; its
    order is then used as given.
    """
    canonical = manifest.split_filenames(split)
    if not canonical:
        raise DataError(f"manifest has no records for split {split!r}")
    if image_list is None:
        return canonical
    given = read_lines(image_list)
    if len(set(given)) != len(given):
        raise DataError(f"{image_list}: duplicate filenames")
    if sorted(given) != canonical:
        extra = sorted(set(given) - set(canonical))[:5]
        absent = sorted(set(canonical) - set(given))[:5]
        raise DataError(f"{image_list} does not match the manifest's {split} split (extra {extra}, absent {absent})")
    return given


def select_ids(order, ids):
    """Keep only the filenames of ``order`` whose image ID is listed in ``ids``, in split order.

    Every listed ID must belong to the split exactly once; used to generate
    features for a named subset (for example records missing from a cache).
    """
    by_id = {image_id_from_filename(fn): fn for fn in order}
    if len(set(ids)) != len(ids):
        raise DataError("duplicate IDs in the requested subset")
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise DataError(f"{len(unknown)} requested IDs are not in this split, e.g. {unknown[:5]}")
    wanted = set(ids)
    return [fn for fn in order if image_id_from_filename(fn) in wanted]
