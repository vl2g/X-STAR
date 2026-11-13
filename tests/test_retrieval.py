"""Retrieval: query and gallery tensors share the requested device; CLI ranking is unchanged."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fixtures import make_layout

import xstar.retrieval as retrieval
from xstar.checkpoints import build_model, load_checkpoint
from xstar.config import PathsConfig
from xstar.data import ImageTokens, OcrBank, PackedQueries
from xstar.utils import sha256_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retrieve.py"


@pytest.fixture
def layout(tmp_path):
    lay = make_layout(tmp_path)
    paths = PathsConfig.load(lay["paths"])
    model, K = build_model(load_checkpoint(lay["checkpoint"]))
    images = ImageTokens(paths.image_tokens("test"), paths.split_list("test"))
    X, Xm = retrieval.build_bank(model, images, images.ids, OcrBank(paths.ocr_dir("test"), K))
    bank = tmp_path / "bank.npz"
    retrieval.save_bank(bank, X, Xm, images.ids, sha256_file(lay["checkpoint"]),
                        {"split": "test", "K": K, "ocr_dir": str(paths.ocr_dir("test"))})
    return {**lay, "paths_obj": paths, "bank": bank, "X": X, "Xm": Xm, "gallery": images.ids}


def _run_cli(argv, capsys):
    old = sys.argv
    sys.argv = [str(SCRIPT)] + argv
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = old
    return json.loads(capsys.readouterr().out)


def _record_devices(monkeypatch):
    seen = []

    def fake_scores(Q, Q_mask, X, X_mask, **_):
        seen.append({t.device.type for t in (Q, Q_mask, X, X_mask)})
        return torch.arange(X.shape[0], dtype=torch.float32).repeat(Q.shape[0], 1)   # CPU stand-in scores

    monkeypatch.setattr(retrieval, "maxsim_scores", fake_scores)
    return seen


def test_rank_moves_numpy_queries_and_cpu_bank_to_requested_device(monkeypatch):
    seen = _record_devices(monkeypatch)
    Q = np.random.default_rng(0).standard_normal((1, 77, 8)).astype(np.float32)
    Qm = np.ones((1, 77), dtype=bool)
    retrieval.rank(Q, Qm, torch.zeros(4, 3, 8), torch.ones(4, 3, dtype=torch.bool), top_k=2, device="meta")
    assert seen == [{"meta"}]


@pytest.mark.parametrize("mode", ["cached", "free_text"])
def test_cli_places_query_and_gallery_on_selected_device(layout, monkeypatch, capsys, mode):
    """Regression for the CPU-query/GPU-gallery mismatch, exercised without CUDA via the meta device."""
    seen = _record_devices(monkeypatch)
    argv = ["--paths", str(layout["paths"]), "--checkpoint", str(layout["checkpoint"]), "--bank-cache",
            str(layout["bank"]), "--device", "meta", "--top-k", "2"]
    if mode == "cached":
        argv += ["--query-image-id", "B_image_8", "--query-set", "hindi_to_english"]
    else:
        import xstar.encoders as enc
        monkeypatch.setattr(enc, "load_clip", lambda *a, **k: (None, None))
        monkeypatch.setattr(enc, "encode_text_tokens", lambda model, texts, device="cpu":
                            (np.ones((1, 77, 512), np.float32), np.ones((1, 77), bool)))
        argv += ["--query", "a shop sign"]
    out = _run_cli(argv, capsys)
    assert seen == [{"meta"}]
    assert len(out["results"]) == 2


def test_cli_cpu_ranking_matches_direct_scoring(layout, capsys):
    out = _run_cli(["--paths", str(layout["paths"]), "--checkpoint", str(layout["checkpoint"]), "--bank-cache",
                    str(layout["bank"]), "--device", "cpu", "--top-k", "5",
                    "--query-image-id", "B_image_8", "--query-set", "hindi_to_english"], capsys)
    pq = PackedQueries(*layout["paths_obj"].packed_queries("test", "hindi_to_english"))
    row = pq.ids.index("B_image_8")
    Q, Qm = pq.single_caption(slice(row, row + 1))
    S = retrieval.maxsim_scores(torch.from_numpy(Q), torch.from_numpy(Qm), layout["X"], layout["Xm"])[0]
    order = torch.sort(S, descending=True, stable=True).indices.tolist()
    assert [r["image_id"] for r in out["results"]] == [layout["gallery"][j] for j in order]
    assert [r["score"] for r in out["results"]] == pytest.approx(S[order].tolist(), abs=1e-6)
    assert out["query_text_encoded"] == "hindi back translated B/image_8.jpg"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available: GPU retrieval NOT exercised")
def test_cuda_retrieval_matches_cpu(layout, capsys):
    common = ["--paths", str(layout["paths"]), "--checkpoint", str(layout["checkpoint"]), "--bank-cache",
              str(layout["bank"]), "--top-k", "5", "--query-image-id", "N_image_10", "--query-set", "english_query"]
    cpu = _run_cli(common + ["--device", "cpu"], capsys)
    gpu = _run_cli(common + ["--device", "cuda:0"], capsys)
    assert [r["image_id"] for r in gpu["results"]] == [r["image_id"] for r in cpu["results"]]
    assert [r["score"] for r in gpu["results"]] == pytest.approx([r["score"] for r in cpu["results"]], abs=1e-4)
    Q = np.ones((1, 77, 512), np.float32)
    top, scores = retrieval.rank(Q, np.ones((1, 77), bool), layout["X"], layout["Xm"], top_k=3, device="cuda:0")
    assert top.device.type == "cuda" and scores.device.type == "cuda"
