"""CLIP encoders reproduce the cached research features (needs local weights and data)."""

import numpy as np
import pytest

from xstar.data import Manifest, load_ocr_embeddings, normalize_ocr_lines, read_lines, split_order


@pytest.fixture(scope="module")
def clip_model():
    pytest.importorskip("clip")
    from xstar.encoders import clip_weights_path, load_clip
    if not clip_weights_path().is_file():
        pytest.skip("ViT-B/32 weights not cached locally (tests never download)")
    return load_clip("cpu")


@pytest.mark.research_data
def test_queries_ocr_and_images_match_cache(research_root, clip_model):
    from xstar.encoders import (
        configure_pillow,
        encode_image_global,
        encode_image_tokens,
        encode_text_global,
        encode_text_tokens,
        load_image,
    )
    model, preprocess = clip_model
    ds = research_root / "XSTAR_dataset"
    emb = ds / "XSTAR_embeddings"
    m = Manifest.load(ds / "XSTAR_final_manifest2.json")
    order = split_order(m, "test")[:2]
    ids = read_lines(emb / "colbert_packed2" / "test_q_ids__hindi_to_english.txt")[:2]
    texts = [m.query_text(i, "hindi_to_english") for i in ids]

    tok, msk = encode_text_tokens(model, texts)
    cached_tok = np.load(emb / "colbert_packed2" / "test_q_tok__hindi_to_english.npy", mmap_mode="r")[:2, 0]
    cached_msk = np.load(emb / "colbert_packed2" / "test_q_mask__hindi_to_english.npy")[:2, 0]
    assert np.array_equal(msk, cached_msk)
    cos = (tok[msk] * np.asarray(cached_tok)[msk]).sum(-1)
    assert cos.min() > 0.9999

    g = encode_text_global(model, texts)
    cached_g = np.load(emb / "to_english_vitb32_2" / "test_hindi_to_english_vitb32.npy")[:2]
    assert (g * cached_g).sum(-1).min() > 0.9999

    lines = normalize_ocr_lines(m.record(ids[0])["ocr_lines_en"])
    E = encode_text_global(model, lines)
    assert (E * load_ocr_embeddings(emb / "ocr_test_vitb32_lines" / f"{ids[0]}.npz")).sum(-1).min() > 0.9999

    import torch
    configure_pillow()
    x = torch.stack([preprocess(load_image(ds / order[0]))])
    t = encode_image_tokens(model, x).numpy()[0]
    cached_t = np.asarray(np.load(emb / "test_image_tokens_postxfmr_vitb32.npy", mmap_mode="r")[0])
    cos_t = (t * cached_t).sum(-1) / np.linalg.norm(t, axis=-1) / np.linalg.norm(cached_t, axis=-1)
    assert cos_t.min() > 0.9999
    gi = encode_image_global(model, x).numpy()[0]
    assert float(gi @ np.load(emb / "test_image_global_embeddings.npy")[0]) > 0.9999
