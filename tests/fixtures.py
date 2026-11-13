"""A tiny synthetic X-STAR layout (manifest, split lists, features, checkpoint) for tests."""

import json
from pathlib import Path

import numpy as np
import torch

from xstar.constants import CLIP_DIM, QUERY_SETS, TEXT_CONTEXT, TRANSLATED_LANGUAGES, VISUAL_TOKENS
from xstar.data import write_ocr_file
from xstar.fusion import FusionLateBank

SPLIT_FILES = {
    "train": ["A/image_1.jpg", "A/image_2.jpg", "B/image_3.jpg", "N/image_4.jpg"],
    "val": ["A/image_5.jpg", "N/image_6.jpg"],
    "test": ["A/image_7.jpg", "B/image_8.jpg", "B/image_9.jpg", "N/image_10.jpg", "N/image_11.jpg"],
}


def _iid(fn):
    folder, name = fn.split("/")
    return f"{folder}_{Path(name).stem}"


def make_layout(root, seed=0, shuffle_test_queries=True):
    root = Path(root)
    rng = np.random.default_rng(seed)
    data, emb = root / "data", root / "features"
    emb.mkdir(parents=True)
    records = []
    for split, files in SPLIT_FILES.items():
        for k, fn in enumerate(files):
            rec = {"image_id": _iid(fn), "filename": fn, "abs_path": "", "language": fn.split("/")[0], "split": split,
                   "english_query": f"english query {fn}", "ocr_lines_en": [f"line {j} of {fn}" for j in range(k % 3)]}
            for lang in TRANSLATED_LANGUAGES:
                rec[f"{lang}_query"] = f"{lang} native {fn}"
                rec[f"{lang}_to_english"] = f"{lang} back translated {fn}"
            records.append(rec)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"meta": {}, "data": records}), encoding="utf-8")

    for split, files in SPLIT_FILES.items():
        order = sorted(files)
        ids = [_iid(f) for f in order]
        (emb / f"image_filenames_{split}.txt").write_text("\n".join(order) + "\n")
        np.save(emb / f"{split}_image_tokens_postxfmr_vitb32.npy",
                rng.standard_normal((len(order), VISUAL_TOKENS, CLIP_DIM)).astype(np.float32))
        g = rng.standard_normal((len(order), CLIP_DIM)).astype(np.float32)
        np.save(emb / f"{split}_image_global_embeddings.npy", g / np.linalg.norm(g, axis=1, keepdims=True))
        ocr = emb / f"ocr_{split}_vitb32_lines"
        ocr.mkdir()
        by_id = {r["image_id"]: r for r in records}
        for iid in ids:
            lines = by_id[iid]["ocr_lines_en"]
            E = rng.standard_normal((len(lines), CLIP_DIM)).astype(np.float32)
            write_ocr_file(ocr / f"{iid}.npz", E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6), lines)
        packed = emb / "colbert_packed2"
        packed.mkdir(exist_ok=True)
        sets = QUERY_SETS if split == "test" else ("english_query",)
        for qs in sets:
            q_ids = list(ids)
            if split == "test" and shuffle_test_queries:
                q_ids = q_ids[::-1]          # query rows deliberately not in gallery order
            suffix = "" if qs == "english_query" else f"__{qs}"
            tok = rng.standard_normal((len(q_ids), 1, TEXT_CONTEXT, CLIP_DIM)).astype(np.float32)
            mask = np.zeros((len(q_ids), 1, TEXT_CONTEXT), dtype=bool)
            mask[:, :, :6] = True
            np.save(packed / f"{split}_q_tok{suffix}.npy", tok)
            np.save(packed / f"{split}_q_mask{suffix}.npy", mask)
            (packed / f"{split}_q_ids{suffix}.txt").write_text("\n".join(q_ids) + "\n")
            if split == "test":
                if qs == "english_query":
                    gd, stem, id_name = emb / "english_query_vitb32_2", "test_english_query", "test_image_ids.txt"
                else:
                    gd, stem, id_name = emb / "to_english_vitb32_2", f"test_{qs}", f"test_{qs}_image_ids.txt"
                gd.mkdir(exist_ok=True)
                gq = rng.standard_normal((len(q_ids), CLIP_DIM)).astype(np.float32)
                np.save(gd / f"{stem}_vitb32.npy", gq / np.linalg.norm(gq, axis=1, keepdims=True))  # L2, as cached
                (gd / id_name).write_text("\n".join(q_ids) + "\n")
                (gd / f"{stem}_strings.txt").write_text("\n".join(by_id[i][qs] for i in q_ids) + "\n")

    ckpt_dir = root / "ckpts"
    ckpt_dir.mkdir()
    torch.manual_seed(seed)
    model = FusionLateBank(d=CLIP_DIM, h=4, L=1, pdrop=0.2, ffn_mult=2, init_tau=1 / 0.03)
    cfg = {"D": CLIP_DIM, "H": 4, "L": 1, "FFN_MULT": 2, "K": 3, "DROP": 0.2}
    torch.save({"epoch": 1, "model": model.state_dict(), "config": cfg, "metrics": {}}, ckpt_dir / "init.pt")

    paths = {
        "dataset": {"root": str(data), "manifest": str(manifest),
                    "split_lists": {s: str(emb / f"image_filenames_{s}.txt") for s in SPLIT_FILES}},
        "features": {"root": str(emb)},
        "checkpoints": {"textcaps_init": str(ckpt_dir / "init.pt")},
    }
    import yaml
    paths_file = root / "paths.yaml"
    paths_file.write_text(yaml.safe_dump(paths))
    return {"root": root, "paths": paths_file, "features": emb, "manifest": manifest,
            "checkpoint": ckpt_dir / "init.pt"}
