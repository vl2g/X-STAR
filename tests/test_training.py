"""Training entry point on a synthetic layout: init versus resume, output safety, schedule."""

import json

import pytest
import torch
from fixtures import make_layout

from xstar.checkpoints import CheckpointError, load_checkpoint
from xstar.config import PathsConfig, StageConfig
from xstar.training import DEFAULT_DATALOADER_TIMEOUT_S, batches, build_loaders, cosine_schedule, train
from xstar.utils import OutputSafetyError


def _stage(**training):
    t = {"init_checkpoint": "textcaps_init", "epochs": 3, "batch_size": 2, "lr": 1e-4, "weight_decay": 1e-4,
         "warmup_steps": 1, "grad_clip_norm": 1.0, "seed": 42, "normalize_queries": True, "maxsim_chunk": 2,
         "num_workers": 0}
    t.update(training)
    return StageConfig({"stage": "test_stage", "data": {"kind": "xstar", "query_set": "english_query"},
                        "model": {"dim": 512, "heads": 4, "layers": 1, "ffn_mult": 2, "dropout": 0.2, "ocr_max_lines": 3},
                        "temperature": {"mode": "stored_parameter", "init_parameter": 50.0}, "training": t})


def test_schedule_warmup_and_cosine():
    assert cosine_schedule(0, 100, 1.0, warmup=10) == 0.0
    assert cosine_schedule(5, 100, 1.0, warmup=10) == pytest.approx(0.5)
    assert cosine_schedule(10, 100, 1.0, warmup=10) == pytest.approx(1.0)
    assert cosine_schedule(100, 100, 1.0, warmup=10) == pytest.approx(0.0)


def test_init_then_resume(tmp_path):
    lay = make_layout(tmp_path / "layout")
    paths = PathsConfig.load(lay["paths"])
    run = tmp_path / "run"
    r1 = train(_stage(), paths, run, init_checkpoint=lay["checkpoint"], max_train_batches=1, max_val_batches=1,
               log=lambda *_: None)
    assert r1["best_epoch"] == 1
    meta = json.loads((run / "run_metadata.json").read_text())
    assert meta["init_checkpoint"]["sha256"] and meta["smoke_run"]
    ck1 = next((run / "checkpoints").glob("epoch01_*.pt"))
    state = load_checkpoint(ck1)
    init_tau = float(load_checkpoint(lay["checkpoint"])["model"]["tau"])
    assert float(state["model"]["tau"]) == pytest.approx(init_tau)          # stored tau inherited, never trained
    assert state["xstar"]["global_step"] == 1 and state["optim"]["state"]   # optimiser state recorded
    assert (run / "checkpoints" / "best.pt").is_file()

    with pytest.raises(OutputSafetyError, match="not empty"):
        train(_stage(), paths, run, init_checkpoint=lay["checkpoint"], log=lambda *_: None)

    train(_stage(), paths, run, resume=ck1, max_train_batches=1, max_val_batches=1, log=lambda *_: None)
    ck2 = next((run / "checkpoints").glob("epoch02_*.pt"))
    s2 = load_checkpoint(ck2)
    assert s2["epoch"] == 2 and s2["xstar"]["global_step"] == 2
    assert (run / "resume_metadata_epoch01.json").is_file()


def test_outputs_inside_inputs_are_refused(tmp_path):
    lay = make_layout(tmp_path / "layout")
    paths = PathsConfig.load(lay["paths"])
    with pytest.raises(OutputSafetyError, match="overlaps protected input"):
        train(_stage(), paths, lay["features"] / "new_run", init_checkpoint=lay["checkpoint"], log=lambda *_: None)
    with pytest.raises(OutputSafetyError):
        train(_stage(), paths, lay["checkpoint"].parent, init_checkpoint=lay["checkpoint"], log=lambda *_: None)


def test_fixed_temperature_stage(tmp_path):
    lay = make_layout(tmp_path / "layout")
    paths = PathsConfig.load(lay["paths"])
    stage = _stage(init_checkpoint=None)
    stage.tau_mode, stage.fixed_multiplier = "fixed", 50.0
    train(stage, paths, tmp_path / "run", max_train_batches=1, max_val_batches=1, log=lambda *_: None)
    state = load_checkpoint(next((tmp_path / "run" / "checkpoints").glob("epoch01_*.pt")))
    assert state["xstar"]["tau_mode"] == "fixed" and state["xstar"]["fixed_multiplier"] == 50.0
    assert float(state["model"]["tau"]) == pytest.approx(50.0)   # init_parameter, unchanged by training
    assert torch.isfinite(torch.tensor(state["metrics"]["train_loss"]))


# ---------------------------------------------------------------------------
# Resume safety: the run definition must not change between epochs.
# ---------------------------------------------------------------------------

def _first_run(tmp_path, stage=None):
    lay = make_layout(tmp_path / "layout")
    paths = PathsConfig.load(lay["paths"])
    run = tmp_path / "run"
    train(stage or _stage(), paths, run, init_checkpoint=lay["checkpoint"], max_train_batches=1, max_val_batches=1,
          log=lambda *_: None)
    return lay, paths, run, next((run / "checkpoints").glob("epoch01_*.pt"))


def _snapshot(run):
    return sorted(str(p.relative_to(run)) for p in run.rglob("*"))


def _set_fixed(stage, mult):
    stage.tau_mode, stage.fixed_multiplier = "fixed", mult


def _alt_ocr_variant(stage, paths, lay):
    import shutil
    for split in ("train", "val"):
        shutil.copytree(lay["features"] / f"ocr_{split}_vitb32_lines", lay["features"] / f"ocr_{split}_alt")
    paths.ocr_dirs["alt"] = "ocr_{split}_alt"
    stage.data["ocr_variant"] = "alt"


def _extra_ocr_file(stage, paths, lay):
    (lay["features"] / "ocr_train_vitb32_lines" / "Z_image_0.npz").write_bytes(b"not part of the run")


CHANGES = {
    "temperature_mode": lambda st, pa, la: _set_fixed(st, 50.0),
    "stage_name": lambda st, pa, la: setattr(st, "stage", "another_stage"),
    "data_query_set_setting": lambda st, pa, la: st.data.update(train_split="val"),
    "data_ocr_variant": _alt_ocr_variant,
    "data_ocr_contents": _extra_ocr_file,
    "architecture_layers": lambda st, pa, la: st.model.update(layers=2),
    "ocr_cap": lambda st, pa, la: st.model.update(ocr_max_lines=4),
    "dropout": lambda st, pa, la: st.model.update(dropout=0.1),
    "schedule_lr": lambda st, pa, la: st.training.update(lr=2e-4),
    "schedule_epochs": lambda st, pa, la: st.training.update(epochs=5),
    "schedule_warmup": lambda st, pa, la: st.training.update(warmup_steps=7),
    "schedule_batch_size": lambda st, pa, la: st.training.update(batch_size=1),
    "seed": lambda st, pa, la: st.training.update(seed=1),
}


@pytest.mark.parametrize("change", sorted(CHANGES))
def test_resume_rejects_changed_run_definition(tmp_path, change):
    lay, paths, run, ck1 = _first_run(tmp_path)
    before = _snapshot(run)
    stage = _stage()
    CHANGES[change](stage, paths, lay)
    with pytest.raises(CheckpointError, match="resume refused"):
        train(stage, paths, run, resume=ck1, max_train_batches=1, max_val_batches=1, log=lambda *_: None)
    assert _snapshot(run) == before            # nothing written


def test_resume_rejects_changed_fixed_multiplier(tmp_path):
    base = _stage()
    _set_fixed(base, 50.0)
    _, paths, run, ck1 = _first_run(tmp_path, base)
    changed = _stage()
    _set_fixed(changed, 40.0)
    with pytest.raises(CheckpointError, match="temperature.fixed_multiplier"):
        train(changed, paths, run, resume=ck1, max_train_batches=1, max_val_batches=1, log=lambda *_: None)
    same = _stage()
    _set_fixed(same, 50.0)
    train(same, paths, run, resume=ck1, max_train_batches=1, max_val_batches=1, log=lambda *_: None)


def test_resume_rejects_changed_smoke_limits(tmp_path):
    _, paths, run, ck1 = _first_run(tmp_path)
    with pytest.raises(CheckpointError, match="limits.max_train_batches"):
        train(_stage(), paths, run, resume=ck1, max_train_batches=2, max_val_batches=1, log=lambda *_: None)


@pytest.mark.worker_processes
def test_resume_allows_documented_benign_changes(tmp_path, worker_processes):
    _, paths, run, ck1 = _first_run(tmp_path)
    train(_stage(num_workers=1, dataloader_timeout_s=60), paths, run, resume=ck1, device="cpu",
          max_train_batches=1, max_val_batches=1, log=lambda *_: None)
    ck2 = load_checkpoint(next((run / "checkpoints").glob("epoch02_*.pt")))
    ck1_state = load_checkpoint(ck1)
    assert ck2["xstar"]["run_signature"] == ck1_state["xstar"]["run_signature"]
    assert ck2["xstar"]["init_checkpoint_sha256"] == ck1_state["xstar"]["init_checkpoint_sha256"] is not None
    assert ck2["xstar"]["global_step"] == 2 and ck2["epoch"] == 2


def test_resume_without_signature_requires_new_run(tmp_path):
    _, paths, run, ck1 = _first_run(tmp_path)
    state = load_checkpoint(ck1)
    del state["xstar"]["run_signature"]
    legacy = run / "checkpoints" / "legacy_epoch01.pt"
    torch.save(state, legacy)
    with pytest.raises(CheckpointError, match="--init-checkpoint"):
        train(_stage(), paths, run, resume=legacy, max_train_batches=1, max_val_batches=1, log=lambda *_: None)


def test_resume_requires_matching_run_metadata(tmp_path):
    _, paths, run, ck1 = _first_run(tmp_path)
    meta = json.loads((run / "run_metadata.json").read_text())
    meta["run_signature"]["stage"] = "edited"
    (run / "run_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="does not match the run signature"):
        train(_stage(), paths, run, resume=ck1, max_train_batches=1, max_val_batches=1, log=lambda *_: None)


# ---------------------------------------------------------------------------
# DataLoader workers: no indefinite waits, no extra batch requests.
# ---------------------------------------------------------------------------

class _Counting(torch.utils.data.Dataset):
    def __init__(self, n=8):
        self.n, self.calls = n, 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.calls += 1
        return torch.full((2,), float(i))


def test_loaders_use_a_batch_timeout_only_with_workers():
    ds = _Counting()
    tr, va = build_loaders(ds, ds, 2, workers=1, device="cpu")
    assert tr.timeout == va.timeout == DEFAULT_DATALOADER_TIMEOUT_S and tr.persistent_workers
    tr0, _ = build_loaders(ds, ds, 2, workers=0, device="cpu")
    assert tr0.timeout == 0


def test_batch_limit_does_not_request_an_extra_batch():
    ds = _Counting()
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    assert len(list(batches(loader, 2))) == 2 and ds.calls == 2
    assert len(list(batches(loader))) == 8


def _deny_shared_memory(_worker_id):
    def refuse(self, *a, **k):
        raise OSError(1, "simulated: shared memory denied")
    torch.UntypedStorage._share_fd_cpu_ = refuse
    torch.UntypedStorage._share_filename_cpu_ = refuse


def test_lost_worker_batch_fails_clearly_instead_of_waiting(capfd):
    """A batch the worker cannot send is dropped by PyTorch; the timeout turns that into an error."""
    import time
    loader = torch.utils.data.DataLoader(_Counting(), batch_size=2, num_workers=1, timeout=3,
                                         worker_init_fn=_deny_shared_memory)
    start = time.time()
    with pytest.raises(RuntimeError, match="--num-workers 0"):
        list(batches(loader, 1))
    assert time.time() - start < 30


def test_disallowed_initialisation_checkpoint_is_refused_before_writing(tmp_path):
    from xstar.utils import sha256_file
    lay = make_layout(tmp_path / "layout")
    paths = PathsConfig.load(lay["paths"])
    stage = _stage(disallowed_init_sha256=[sha256_file(lay["checkpoint"]).upper()])
    with pytest.raises(CheckpointError, match="disallowed_init_sha256"):
        train(stage, paths, tmp_path / "run", init_checkpoint=lay["checkpoint"], log=lambda *_: None)
    assert not (tmp_path / "run").exists()
    ok = _stage(disallowed_init_sha256=["0" * 64])                # other checkpoints remain usable
    train(ok, paths, tmp_path / "run_ok", init_checkpoint=lay["checkpoint"], max_train_batches=1, max_val_batches=1,
          log=lambda *_: None)
