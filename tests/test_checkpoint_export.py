"""Inference-only checkpoint export: exact round trip, metadata, refusals and training interaction."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fixtures import make_layout

from xstar.checkpoints import (
    INFERENCE_FORMAT,
    CheckpointError,
    build_model,
    compare_state_dicts,
    export_inference_checkpoint,
    is_inference_only,
    load_checkpoint,
)
from xstar.config import PathsConfig, StageConfig
from xstar.training import train
from xstar.utils import sha256_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_inference_checkpoint.py"


def _stage(**training):
    t = {"init_checkpoint": "textcaps_init", "epochs": 1, "batch_size": 2, "lr": 1e-4, "weight_decay": 1e-4,
         "warmup_steps": 1, "grad_clip_norm": 1.0, "seed": 42, "normalize_queries": True, "maxsim_chunk": 2,
         "num_workers": 0}
    t.update(training)
    return StageConfig({"stage": "test_stage", "data": {"kind": "xstar", "query_set": "english_query"},
                        "model": {"dim": 512, "heads": 4, "layers": 1, "ffn_mult": 2, "dropout": 0.2, "ocr_max_lines": 3},
                        "temperature": {"mode": "stored_parameter", "init_parameter": 50.0}, "training": t})


@pytest.fixture
def trained(tmp_path):
    """A checkpoint written by the trainer (optimiser, RNG state, run signature) plus its layout."""
    lay = make_layout(tmp_path / "layout")
    paths = PathsConfig.load(lay["paths"])
    run = tmp_path / "run"
    train(_stage(), paths, run, init_checkpoint=lay["checkpoint"], max_train_batches=1, max_val_batches=1,
          log=lambda *_: None)
    return {"layout": lay, "paths": paths, "run": run, "ckpt": next((run / "checkpoints").glob("epoch01_*.pt"))}


def test_round_trip_is_exact_and_drops_training_state(trained, tmp_path):
    src_path, out = trained["ckpt"], tmp_path / "inference.pt"
    src = load_checkpoint(src_path)
    summary = export_inference_checkpoint(src_path, out, experiment="unit", split="synthetic")
    exp = load_checkpoint(out)

    assert compare_state_dicts(src["model"], exp["model"]) == []
    assert [k for k in src["model"]] == [k for k in exp["model"]]
    assert all(src["model"][k].dtype == exp["model"][k].dtype for k in src["model"])
    assert exp["config"] == src["config"] and exp["epoch"] == src["epoch"] and exp["metrics"] == src["metrics"]
    assert "optim" not in exp and "rng" not in exp["xstar"] and "run_signature" not in exp["xstar"]
    meta = exp["xstar"]
    assert meta["format"] == INFERENCE_FORMAT and meta["inference_only"] and is_inference_only(exp)
    assert meta["tau_mode"] == "stored_parameter" and meta["fixed_multiplier"] is None
    assert meta["stored_tau"] == pytest.approx(float(src["model"]["tau"]))
    assert meta["ocr_max_lines"] == src["config"]["K"] == 3
    assert meta["inference"]["bank_tokens"] == 50 + 3
    assert meta["source"]["sha256"] == sha256_file(src_path) and meta["source"]["filename"] == src_path.name
    assert meta["source"]["experiment"] == "unit" and meta["source"]["split"] == "synthetic"
    assert meta["source"]["training_metadata"]["global_step"] == 1
    assert set(meta["removed_from_source"]) == {"optim", "xstar.rng", "xstar.run_signature"}
    assert set(meta["exporter"]) >= {"code_revision", "xstar_version", "torch_version", "created_utc"}
    assert summary["output"]["size_bytes"] < summary["source"]["size_bytes"]
    assert str(tmp_path) not in repr(meta)                  # no machine paths in the artifact

    # Same outputs from the supported loader: fused bank and temperature.
    torch.manual_seed(0)
    V, O = torch.randn(2, 50, 512), torch.randn(2, 3, 512)
    mO = torch.tensor([[True, False, False], [True, True, True]])
    m_src, K_src = build_model(src)
    m_exp, K_exp = build_model(exp)
    m_src.eval(), m_exp.eval()
    with torch.no_grad():
        Xa, Ma, ta = m_src(V, O, mask_O=mO)
        Xb, Mb, tb = m_exp(V, O, mask_O=mO)
    assert K_src == K_exp and torch.equal(Xa, Xb) and torch.equal(Ma, Mb) and torch.equal(ta, tb)


def test_original_style_checkpoint_needs_explicit_temperature(trained, tmp_path):
    original = trained["layout"]["checkpoint"]              # no 'xstar' metadata, like the research checkpoints
    with pytest.raises(CheckpointError, match="no temperature metadata"):
        export_inference_checkpoint(original, tmp_path / "a.pt", experiment="e", split="s")
    assert not (tmp_path / "a.pt").exists()
    with pytest.raises(CheckpointError, match="needs --fixed-multiplier"):
        export_inference_checkpoint(original, tmp_path / "a.pt", experiment="e", split="s", tau_mode="fixed")

    export_inference_checkpoint(original, tmp_path / "fixed.pt", experiment="e", split="s", tau_mode="fixed",
                                fixed_multiplier=50.0)
    exp = load_checkpoint(tmp_path / "fixed.pt")
    assert exp["xstar"]["tau_mode"] == "fixed" and exp["xstar"]["fixed_multiplier"] == 50.0
    assert exp["xstar"]["tau_mode_origin"] == "stated at export"
    model, _ = build_model(exp)
    assert float(model.temperature()) == 50.0               # multiplier, not the stored parameter
    assert float(model.tau) == pytest.approx(float(load_checkpoint(original)["model"]["tau"]))


def test_recorded_temperature_cannot_be_overridden(trained, tmp_path):
    with pytest.raises(CheckpointError, match="refusing to export it as 'fixed'"):
        export_inference_checkpoint(trained["ckpt"], tmp_path / "x.pt", experiment="e", split="s",
                                    tau_mode="fixed", fixed_multiplier=50.0)
    assert not (tmp_path / "x.pt").exists()


def test_overwrite_and_reexport_are_refused(trained, tmp_path):
    out = tmp_path / "inference.pt"
    out.write_bytes(b"keep me")
    with pytest.raises(CheckpointError, match="refusing to overwrite"):
        export_inference_checkpoint(trained["ckpt"], out, experiment="e", split="s")
    assert out.read_bytes() == b"keep me"

    good = tmp_path / "good.pt"
    export_inference_checkpoint(trained["ckpt"], good, experiment="e", split="s")
    with pytest.raises(CheckpointError, match="already an inference-only export"):
        export_inference_checkpoint(good, tmp_path / "again.pt", experiment="e", split="s")


def test_unrecognised_entries_are_not_dropped_silently(trained, tmp_path):
    state = load_checkpoint(trained["ckpt"])
    state["ema_model"] = {"w": torch.zeros(1)}
    odd = tmp_path / "odd.pt"
    torch.save(state, odd)
    with pytest.raises(CheckpointError, match="unrecognised top-level entries"):
        export_inference_checkpoint(odd, tmp_path / "o.pt", experiment="e", split="s")


def test_inference_export_cannot_resume_but_can_initialise(trained, tmp_path):
    exp = trained["run"] / "checkpoints" / "inference.pt"   # inside the run directory, as resume requires
    export_inference_checkpoint(trained["ckpt"], exp, experiment="e", split="s")
    with pytest.raises(CheckpointError, match="inference-only export .* cannot resume"):
        train(_stage(), trained["paths"], trained["run"], resume=exp, max_train_batches=1, max_val_batches=1,
              log=lambda *_: None)
    # A new run may start from its weights.
    train(_stage(), trained["paths"], tmp_path / "new_run", init_checkpoint=exp, max_train_batches=1,
          max_val_batches=1, log=lambda *_: None)


def test_export_of_a_disallowed_checkpoint_is_still_refused_as_init(trained, tmp_path):
    exp = tmp_path / "inference.pt"
    export_inference_checkpoint(trained["ckpt"], exp, experiment="e", split="s")
    stage = _stage(disallowed_init_sha256=[sha256_file(trained["ckpt"])])
    with pytest.raises(CheckpointError, match="inference export of .*disallowed_init_sha256"):
        train(stage, trained["paths"], tmp_path / "new_run", init_checkpoint=exp, max_train_batches=1,
              max_val_batches=1, log=lambda *_: None)
    assert not (tmp_path / "new_run").exists()


def test_script_writes_export_and_report(trained, tmp_path):
    out, report = tmp_path / "cli.pt", tmp_path / "cli.json"
    cmd = [sys.executable, str(SCRIPT), "--input", str(trained["ckpt"]), "--output", str(out),
           "--experiment", "cli", "--split", "synthetic", "--report", str(report)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert out.is_file() and report.is_file() and "model tensors identical" in r.stdout
    again = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert again.returncode == 2 and again.stderr.startswith("error:") and "Traceback" not in again.stderr
