"""Training loop for both CoLITE-CLIP stages (TextCaps pre-training, X-STAR fine-tuning).

Recovered from the TextCaps pre-training cell (server 40) and main-result
notebook cell 1. Preserved: AdamW over all model parameters, linear warm-up
then cosine decay set per step, gradient clipping at 1.0, symmetric InfoNCE
averaged over the caption axis, validation loss every epoch, and selection of
``best.pt`` by lowest validation loss.

Engineering changes: MaxSim is computed in vectorised chunks instead of Python
loops; initialisation loads weights with ``strict=True`` (the source used
``strict=False`` and reported no missing or unexpected keys); resume restores
optimiser, epoch, step, best value and RNG state, but only after checking that
the run definition (``run_signature``: stage, data settings and input identity,
architecture, OCR cap, temperature, schedule-affecting settings and smoke
limits) is unchanged; only the device and DataLoader worker count may differ.
All outputs go to a new run directory.
"""

import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xstar.checkpoints import (
    CheckpointError,
    checkpoint_identity,
    is_inference_only,
    load_checkpoint,
    save_checkpoint,
    source_sha256,
)
from xstar.data import (
    ImageTokens,
    OcrBank,
    PackedQueries,
    PackedQueryRows,
    PairDataset,
    TextCapsCaptionShards,
    collate_pairs,
)
from xstar.fusion import FusionLateBank
from xstar.losses import multi_caption_nce
from xstar.scoring import maxsim_scores
from xstar.utils import environment_info, prepare_output_dir, sha256_file, utc_now, write_json_new


def cosine_schedule(step, total, base_lr, warmup=0):
    """Linear warm-up then cosine decay (identical in both sources within training)."""
    if step < warmup:
        return base_lr * float(step) / max(1, warmup)
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    np_state = np.random.get_state()
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             # uint32 key stored as int64: torch cannot serialise uint32 tensors.
             "numpy": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), np_state[2], np_state[3],
                       np_state[4])}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), n[2], n[3], n[4]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_datasets(stage, paths):
    """Training and validation datasets for the configured stage."""
    K = int(stage.model["ocr_max_lines"])
    if stage.kind == "xstar":
        query_set = stage.data.get("query_set", "english_query")
        variant = stage.data.get("ocr_variant", paths.ocr_variant)
        out = []
        for split in (stage.data.get("train_split", "train"), stage.data.get("val_split", "val")):
            images = ImageTokens(paths.image_tokens(split), paths.split_list(split))
            queries = PackedQueryRows(PackedQueries(*paths.packed_queries(split, query_set)))
            out.append(PairDataset(images, OcrBank(paths.ocr_dir(split, variant), K), queries, split))
        return out
    tc = paths.textcaps
    if not tc:
        raise ValueError(f"{paths.source}: the TextCaps stage needs a 'textcaps' section")
    train_images = ImageTokens(tc["image_tokens"], tc["image_list"])
    val_images = ImageTokens(tc["val_image_tokens"], tc["image_list"]) if tc.get("val_image_tokens") else train_images
    out = []
    for split, images in (("train", train_images), ("val", val_images)):
        cap = tc["captions"][split]
        shards = TextCapsCaptionShards(cap["index"], cap["rowmap"], cap["ids"], shard_dir=cap.get("shard_dir"))
        out.append(PairDataset(images, OcrBank(tc["ocr_dirs"][split], K), shards, split))
    return out


def checkpoint_config(stage):
    m, t = stage.model, stage.training
    return {"D": m["dim"], "H": m["heads"], "FFN_MULT": m["ffn_mult"], "L": m["layers"], "K": m["ocr_max_lines"],
            "BATCH_SIZE": t["batch_size"], "LR": t["lr"], "WD": t["weight_decay"], "DROP": m["dropout"],
            "EPOCHS": t["epochs"], "WARMUP_STEPS": t["warmup_steps"], "MAX_CHUNK_B": t.get("maxsim_chunk"),
            "SEED": t["seed"]}


# Settings compared on resume. Everything here changes what is trained or how
# the optimiser/schedule evolve; a change requires a new run (initialised from
# weights with --init-checkpoint) instead of a resume.
SCHEDULE_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "warmup_steps", "grad_clip_norm", "seed",
                 "normalize_queries", "maxsim_chunk", "fused_adamw", "allow_tf32")
# The only settings allowed to differ on resume: they affect speed, placement
# or failure detection, not the experiment definition.
RESUME_BENIGN = ("device", "num_workers", "dataloader_timeout_s")
# Seconds a training loop waits for one batch from DataLoader worker processes
# before failing (training.dataloader_timeout_s). Without a limit, a batch lost
# in a worker (for example when the environment denies the shared memory used
# to pass tensors) leaves the main process waiting forever.
DEFAULT_DATALOADER_TIMEOUT_S = 600.0
SIGNATURE_VERSION = 1


def _array_identity(path):
    """Shape, dtype and size of a .npy file (large arrays are not hashed)."""
    path = Path(path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "size_bytes": path.stat().st_size}


def _listing_identity(directory):
    """File count and a hash of sorted (name, size) pairs of a feature directory."""
    entries = sorted((p.name, p.stat().st_size) for p in Path(directory).iterdir() if p.is_file())
    digest = hashlib.sha256("\n".join(f"{n}\t{z}" for n, z in entries).encode("utf-8")).hexdigest()
    return {"files": len(entries), "names_sizes_sha256": digest}


def data_identity(stage, paths):
    """Content identity of the training inputs: hashes of ID/list/index files,
    shapes and sizes of large arrays, and listings of OCR directories."""
    if stage.kind == "xstar":
        query_set = stage.data.get("query_set", "english_query")
        variant = stage.data.get("ocr_variant", paths.ocr_variant)
        ident = {"query_set": query_set, "ocr_variant": variant}
        for role, split in (("train", stage.data.get("train_split", "train")), ("val", stage.data.get("val_split", "val"))):
            tok, mask, ids = paths.packed_queries(split, query_set)
            ident[role] = {"split": split, "split_list_sha256": sha256_file(paths.split_list(split)),
                           "image_tokens": _array_identity(paths.image_tokens(split)),
                           "query_tokens": _array_identity(tok), "query_mask": _array_identity(mask),
                           "query_ids_sha256": sha256_file(ids),
                           "ocr": _listing_identity(paths.ocr_dir(split, variant))}
        return ident
    tc = paths.textcaps
    ident = {"image_tokens": _array_identity(tc["image_tokens"]), "image_list_sha256": sha256_file(tc["image_list"]),
             "val_image_tokens": _array_identity(tc["val_image_tokens"]) if tc.get("val_image_tokens") else None}
    for split in ("train", "val"):
        cap = tc["captions"][split]
        ident[split] = {"caption_index_sha256": sha256_file(cap["index"]), "caption_rowmap_sha256": sha256_file(cap["rowmap"]),
                        "caption_ids_sha256": sha256_file(cap["ids"]), "ocr": _listing_identity(tc["ocr_dirs"][split])}
    return ident


def run_signature(stage, paths, max_train_batches=None, max_val_batches=None):
    """Everything that defines a training run and must not change on resume."""
    t = stage.training
    sig = {"signature_version": SIGNATURE_VERSION, "stage": stage.stage, "data": stage.data,
           "model": stage.model,
           "temperature": {"mode": stage.tau_mode, "fixed_multiplier": stage.fixed_multiplier,
                           "init_parameter": stage.init_tau},
           "schedule": {k: t.get(k) for k in SCHEDULE_KEYS},
           "limits": {"max_train_batches": max_train_batches, "max_val_batches": max_val_batches},
           "inputs": data_identity(stage, paths)}
    return json.loads(json.dumps(sig, sort_keys=True))


def signature_differences(saved, current, prefix=""):
    """Human-readable list of settings that differ between two signatures."""
    if isinstance(saved, dict) and isinstance(current, dict):
        diffs = []
        for k in sorted(set(saved) | set(current)):
            diffs += signature_differences(saved.get(k), current.get(k), f"{prefix}{k}.")
        return diffs
    return [] if saved == current else [f"{prefix.rstrip('.')}: saved {saved!r}, now {current!r}"]


def check_resume(resume, out, signature):
    """Load a resume checkpoint and refuse it unless the run definition is unchanged."""
    state = load_checkpoint(resume)
    meta = state.get("xstar") or {}
    if is_inference_only(state):
        raise CheckpointError(f"{resume} is an inference-only export (no optimiser state, RNG state or run "
                              "signature) and cannot resume training; resume from the original training "
                              "checkpoint, or start a new run with --init-checkpoint")
    if "global_step" not in meta:
        raise CheckpointError(f"{resume} was not written by this trainer; start a new run with --init-checkpoint")
    saved = meta.get("run_signature")
    if saved is None:
        raise CheckpointError(f"{resume} has no run signature (written before resume checks existed), so an exact "
                              "resume cannot be verified; start a new run with --init-checkpoint instead")
    run_meta_path = out / "run_metadata.json"
    if not run_meta_path.is_file():
        raise CheckpointError(f"{run_meta_path} is missing; cannot confirm this checkpoint belongs to the run")
    run_sig = json.loads(run_meta_path.read_text(encoding="utf-8")).get("run_signature")
    if run_sig != saved:
        raise CheckpointError(f"{resume} does not match the run signature in {run_meta_path}")
    diffs = signature_differences(saved, signature)
    if diffs:
        raise CheckpointError("resume refused: the run definition changed (only " + ", ".join(RESUME_BENIGN)
                              + " may differ). Start a new run with --init-checkpoint to change it.\n  "
                              + "\n  ".join(diffs))
    return state, meta


def check_init_allowed(training, init_checkpoint):
    """Refuse an initialisation checkpoint listed in ``training.disallowed_init_sha256``.

    This guard can reject initialization from a model trained on evaluation
    records. An inference export carries its source checkpoint's SHA-256,
    which is checked too; exporting does not bypass the exclusion list.
    """
    disallowed = {str(x).lower() for x in (training.get("disallowed_init_sha256") or [])}
    if init_checkpoint is None or not disallowed:
        return
    digest = sha256_file(init_checkpoint)
    if digest in disallowed:
        raise CheckpointError(f"{init_checkpoint} (SHA-256 {digest[:12]}) is listed in "
                              "training.disallowed_init_sha256 for this stage and cannot initialise it")
    origin = source_sha256(load_checkpoint(init_checkpoint))
    if origin is not None and str(origin).lower() in disallowed:
        raise CheckpointError(f"{init_checkpoint} is an inference export of {str(origin)[:12]}, which is listed in "
                              "training.disallowed_init_sha256 for this stage and cannot initialise it")


def build_loaders(train_ds, val_ds, batch_size, workers, device, timeout_s=DEFAULT_DATALOADER_TIMEOUT_S):
    """Training (shuffled, drop_last) and validation loaders; a batch timeout applies when workers > 0."""
    kw = {"num_workers": workers, "pin_memory": device.startswith("cuda"), "collate_fn": collate_pairs,
          "persistent_workers": workers > 0, "timeout": float(timeout_s) if workers > 0 else 0}
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **kw))


def batches(loader, limit=None):
    """Yield at most ``limit`` batches without requesting an extra one, turning a
    worker timeout into an error that names the likely cause."""
    it = iter(loader)
    count = 0
    while limit is None or count < limit:
        try:
            batch = next(it)
        except StopIteration:
            return
        except RuntimeError as e:
            if "DataLoader timed out" not in str(e):
                raise
            raise RuntimeError(f"{e}: a DataLoader worker process returned no batch. Worker processes may be "
                               "unable to pass tensors here (for example shared memory denied in a sandbox); "
                               "rerun with --num-workers 0 or check tools/check_dataloader_workers.py") from e
        count += 1
        yield batch


def train(stage, paths, output_dir, device="cpu", init_checkpoint=None, resume=None, max_train_batches=None,
          max_val_batches=None, num_workers=None, log=print):
    t = stage.training
    check_init_allowed(t, init_checkpoint)                    # before anything is read or written
    signature = run_signature(stage, paths, max_train_batches, max_val_batches)
    resume = Path(resume).resolve() if resume else None
    resumed = None
    if resume is not None:
        out = Path(os.path.realpath(output_dir))
        if out not in resume.parents:
            raise CheckpointError(f"--resume checkpoint must lie inside the run directory {out}")
        resumed = check_resume(resume, out, signature)          # validated before anything is written
    out = prepare_output_dir(output_dir, paths.protected_roots(), allow_existing=resume is not None)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    set_seed(int(t["seed"]))
    if device.startswith("cuda") and t.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds = build_datasets(stage, paths)
    workers = int(t.get("num_workers", 0) if num_workers is None else num_workers)
    train_loader, val_loader = build_loaders(train_ds, val_ds, int(t["batch_size"]), workers, device,
                                             float(t.get("dataloader_timeout_s", DEFAULT_DATALOADER_TIMEOUT_S)))

    model = FusionLateBank(**stage.model_kwargs()).to(device)
    adamw_kw = {"lr": float(t["lr"]), "weight_decay": float(t["weight_decay"])}
    if t.get("fused_adamw") and device.startswith("cuda"):
        adamw_kw["fused"] = True
    optim = torch.optim.AdamW(model.parameters(), **adamw_kw)

    start_epoch, global_step, best_val, best_epoch = 0, 0, float("inf"), None
    init_info = None
    init_sha = None
    if resumed is not None:
        state, meta = resumed
        init_sha = meta.get("init_checkpoint_sha256")
        model.load_state_dict(state["model"], strict=True)
        optim.load_state_dict(state["optim"])
        start_epoch, global_step = int(state["epoch"]), int(meta["global_step"])
        best_val, best_epoch = float(meta["best_val_loss"]), meta.get("best_epoch")
        restore_rng_state(meta["rng"])
        log(f"Resumed from {resume.name}: epoch {start_epoch}, step {global_step}")
    elif init_checkpoint is not None:
        init = load_checkpoint(init_checkpoint)
        model.load_state_dict(init["model"], strict=True)
        init_info = {**checkpoint_identity(init_checkpoint), "epoch": init.get("epoch"), "config": init.get("config")}
        init_sha = init_info["sha256"]
        log(f"Initialised weights from {init_checkpoint} (stored tau {float(model.tau):.4f}); new optimiser and schedule")

    smoke = max_train_batches is not None or max_val_batches is not None
    run_meta = {"created_utc": utc_now(), "stage": stage.stage, "stage_config": stage.raw,
                "stage_config_file": stage.source, "paths_config_file": paths.source, "device": device,
                "num_workers": workers, "init_checkpoint": init_info, "resumed_from": str(resume) if resume else None,
                "smoke_run": smoke, "max_train_batches": max_train_batches, "max_val_batches": max_val_batches,
                "train_examples": len(train_ds), "val_examples": len(val_ds), "steps_per_epoch": len(train_loader),
                "selection_metric": "lowest validation loss", "run_signature": signature,
                "resume_benign_settings": list(RESUME_BENIGN), "environment": environment_info()}
    write_json_new(out / ("run_metadata.json" if resume is None else f"resume_metadata_epoch{start_epoch:02d}.json"),
                   run_meta)

    epochs = int(t["epochs"])
    total_steps = epochs * max(1, len(train_loader))
    chunk = int(t.get("maxsim_chunk") or t["batch_size"])
    normalize_q = bool(t.get("normalize_queries", False))
    clip_norm = float(t["grad_clip_norm"])

    def batch_loss(batch):
        V = batch["V"].to(device, non_blocking=True)
        O = batch["O"].to(device, non_blocking=True)
        mO = batch["mO"].to(device, non_blocking=True)
        Q = batch["Q"].to(device, non_blocking=True)
        Qm = batch["Qm"].to(device, non_blocking=True)
        X, mX, tau = model(V, O, mask_V=None, mask_O=mO)
        score = lambda q, m: maxsim_scores(q, m, X, mX, query_chunk=q.shape[0], gallery_chunk=chunk,
                                           normalize_queries=normalize_q)
        return multi_caption_nce(score, Q, Qm, float(tau))

    for epoch in range(start_epoch, epochs):
        model.train()
        running, steps = 0.0, 0
        for batch in batches(train_loader, max_train_batches):
            lr = cosine_schedule(global_step, total_steps, float(t["lr"]), warmup=int(t["warmup_steps"]))
            for g in optim.param_groups:
                g["lr"] = lr
            loss = batch_loss(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optim.step()
            running += float(loss.item())
            steps += 1
            global_step += 1
        train_loss = running / max(1, steps)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in batches(val_loader, max_val_batches):
                val_losses.append(float(batch_loss(batch)))
        model.train()
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        improved = not math.isnan(val_loss) and val_loss < best_val
        if improved:
            best_val, best_epoch = val_loss, epoch + 1
        payload = {"epoch": epoch + 1, "model": model.state_dict(), "optim": optim.state_dict(),
                   "config": checkpoint_config(stage), "metrics": {"train_loss": train_loss, "val_loss": val_loss},
                   "xstar": {"stage": stage.stage, "tau_mode": model.tau_mode,
                             "fixed_multiplier": model.fixed_multiplier, "global_step": global_step,
                             "best_val_loss": best_val, "best_epoch": best_epoch, "smoke_run": smoke,
                             "init_checkpoint_sha256": init_sha, "run_signature": signature,
                             "rng": rng_state()}}
        path = save_checkpoint(ckpt_dir / f"epoch{epoch + 1:02d}_train{train_loss:.4f}_val{val_loss:.4f}.pt", payload)
        if improved:
            tmp = ckpt_dir / "best.pt.partial"
            shutil.copyfile(path, tmp)
            os.replace(tmp, ckpt_dir / "best.pt")
        record = {"epoch": epoch + 1, "global_step": global_step, "train_loss": train_loss, "val_loss": val_loss,
                  "lr_last": lr if steps else None, "best_epoch": best_epoch, "checkpoint": path.name}
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        log(f"Epoch {epoch + 1}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | {path.name}")
        if smoke:
            log("Smoke run: stopping after one epoch")
            break
    return {"output_dir": str(out), "best_epoch": best_epoch, "best_val_loss": best_val}
