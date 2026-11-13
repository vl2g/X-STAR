"""The package model reproduces the original research classes and loads their checkpoints."""

import numpy as np
import pytest
import reference_source as ref
import torch

from xstar.checkpoints import build_model, load_checkpoint
from xstar.fusion import FusionLateBank
from xstar.losses import symmetric_nce
from xstar.scoring import maxsim_scores

SMALL = dict(d=32, h=4, L=2, pdrop=0.2, ffn_mult=2)


def _inputs(B=4, Nv=5, K=6, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(B, Nv, d, generator=g)
    O = torch.nn.functional.normalize(torch.randn(B, K, d, generator=g), dim=-1)
    mO = torch.zeros(B, K, dtype=torch.bool)
    mO[0, :] = True          # full OCR
    mO[1, :3] = True         # partial OCR
    mO[3, :1] = True         # one line; image 2 has no OCR at all
    return V, O, mO


@pytest.mark.parametrize("ref_cls", [ref.XStarFusionLateBank, ref.EvalFusionLateBank, ref.TextCapsFusionLateBank])
def test_state_dict_keys_and_outputs_match_source(ref_cls):
    torch.manual_seed(1)
    source = ref_cls(SMALL["d"], SMALL["h"], SMALL["L"], SMALL["pdrop"], SMALL["ffn_mult"]).eval()
    ours = FusionLateBank(**SMALL).eval()
    assert list(ours.state_dict()) == list(source.state_dict())
    ours.load_state_dict(source.state_dict(), strict=True)
    V, O, mO = _inputs()
    with torch.no_grad():
        Xs, Ms, _ = source(V, O, mask_V=None, mask_O=mO)
        Xo, Mo, _ = ours(V, O, mask_V=None, mask_O=mO)
    assert torch.equal(Ms, Mo) and Mo.all()
    torch.testing.assert_close(Xo, Xs, rtol=0, atol=1e-6)


def test_temperature_modes_reproduce_both_sources():
    torch.manual_seed(2)
    textcaps = ref.TextCapsFusionLateBank(32, 4, 1, 0.1, 2).eval()
    xstar_src = ref.XStarFusionLateBank(32, 4, 1, 0.2, 2).eval()
    xstar_src.load_state_dict(textcaps.state_dict())        # source initialisation path
    V, O, mO = _inputs()
    fixed = FusionLateBank(32, 4, 1, 0.1, 2, tau_mode="fixed", fixed_multiplier=50.0).eval()
    fixed.load_state_dict(textcaps.state_dict())
    stored = FusionLateBank(32, 4, 1, 0.2, 2, tau_mode="stored_parameter").eval()
    stored.load_state_dict(textcaps.state_dict())
    with torch.no_grad():
        assert float(textcaps(V, O, None, mO)[2]) == pytest.approx(50.0)
        assert float(fixed(V, O, None, mO)[2]) == pytest.approx(50.0)
        assert float(xstar_src(V, O, None, mO)[2]) == pytest.approx(1 / 0.03)
        assert float(stored(V, O, None, mO)[2]) == pytest.approx(1 / 0.03)
    assert float(fixed.tau) == pytest.approx(1 / 0.03)     # stored value untouched
    assert not stored.tau.requires_grad


def test_null_ocr_token_fills_padding_and_empty_ocr():
    model = FusionLateBank(d=8, h=2, L=0, pdrop=0.0, ffn_mult=2).eval()   # no blocks: bank = normalize([V; O])
    V = torch.randn(2, 3, 8)
    O = torch.randn(2, 4, 8)
    mO = torch.tensor([[True, True, False, False], [False, False, False, False]])
    with torch.no_grad():
        X, mX, _ = model(V, O, mask_V=None, mask_O=mO)
    null = torch.nn.functional.normalize(model.null_O.detach().view(-1), dim=0, eps=1e-6)
    assert mX.all() and X.shape == (2, 7, 8)
    torch.testing.assert_close(X[0, 3], torch.nn.functional.normalize(O[0, 0], dim=0, eps=1e-6))
    for b, k in [(0, 2), (0, 3), (1, 0), (1, 3)]:
        torch.testing.assert_close(X[b, 3 + k], null)


def test_maxsim_and_loss_match_training_source():
    torch.manual_seed(3)
    source = ref.XStarFusionLateBank(32, 4, 2, 0.0, 2).eval()
    ours = FusionLateBank(32, 4, 2, 0.0, 2).eval()
    ours.load_state_dict(source.state_dict())
    V, O, mO = _inputs(B=5)
    Q = torch.randn(5, 9, 32)
    Qm = torch.rand(5, 9) > 0.3
    Qm[:, 0] = True
    with torch.no_grad():
        X, mX, tau = source(V, O, None, mO)
        S_src = ref.maxsim_scores_batch(Q, Qm, X, mX, chunk_b=2, l2norm_q=True)
        S_new = maxsim_scores(Q, Qm, X, mX, query_chunk=2, gallery_chunk=3, normalize_queries=True)
    torch.testing.assert_close(S_new, S_src, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(symmetric_nce(S_new, float(tau)), ref.symmetric_nce(S_src, float(tau)), rtol=1e-5, atol=1e-5)


@pytest.mark.research_data
def test_main_checkpoint_loads_and_matches_source_on_real_features(research_root):
    emb = research_root / "XSTAR_dataset" / "XSTAR_embeddings"
    ckpt = load_checkpoint(emb / "checkpoints_colbert_XSTAR_5" / "best.pt")
    assert ckpt["epoch"] == 9
    model, K = build_model(ckpt)
    model.eval()
    assert K == 40 and len(model.blocks) == 11 and model.blocks[0].attn_vo.n_heads == 4
    assert model.tau_mode == "stored_parameter" and float(model.tau) == pytest.approx(33.3333, abs=1e-3)
    cfg = ckpt["config"]
    source = ref.EvalFusionLateBank(cfg["D"], cfg["H"], cfg["L"], cfg["DROP"], cfg["FFN_MULT"]).eval()
    source.load_state_dict(ckpt["model"], strict=True)

    from xstar.data import ImageTokens, OcrBank, PackedQueries
    images = ImageTokens(emb / "test_image_tokens_postxfmr_vitb32.npy", emb / "image_filenames_test.txt")
    ids = images.ids[:6]
    ocr = OcrBank(emb / "ocr_test_vitb32_lines", K)
    V = torch.from_numpy(np.stack([images.get(i) for i in range(6)]))
    O, mO = map(lambda a: torch.from_numpy(np.stack(a)), zip(*(ocr.get(i) for i in ids)))
    with torch.no_grad():
        Xs, Ms, _ = source(V, O, None, mO)
        Xo, Mo, _ = model(V, O, None, mO)
    torch.testing.assert_close(Xo, Xs, rtol=0, atol=1e-6)
    pq = PackedQueries(emb / "colbert_packed2/test_q_tok.npy", emb / "colbert_packed2/test_q_mask.npy",
                       emb / "colbert_packed2/test_q_ids.txt")
    Q, Qm = pq.single_caption(slice(0, 6))
    Q, Qm = torch.from_numpy(Q), torch.from_numpy(Qm)
    from xstar.scoring import maxsim_reference
    torch.testing.assert_close(maxsim_scores(Q, Qm, Xo, Mo), maxsim_reference(Q, Qm, Xs, Ms), rtol=1e-5, atol=1e-4)
