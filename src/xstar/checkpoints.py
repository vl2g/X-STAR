"""Loading and saving CoLITE-CLIP checkpoints.

Checkpoints are dictionaries with ``model`` (state dict), ``config`` (the
upper-case keys used by the research code: D, H, L, K, FFN_MULT, DROP, ...),
``epoch``, ``metrics`` and, for training checkpoints, ``optim``. Files written
by this package add ``xstar`` metadata (temperature mode, global step, RNG
state, provenance). All loading uses ``torch.load(weights_only=True)``.

Inference-only exports (:func:`export_inference_checkpoint`, format
``xstar-inference-v1``) keep ``model``, ``config``, ``epoch`` and ``metrics``
unchanged and replace the training metadata with the temperature settings,
inference settings and source provenance. They load with the same functions
but cannot resume training.
"""

from pathlib import Path

import torch

from xstar.errors import CheckpointError
from xstar.fusion import FusionLateBank
from xstar.utils import sha256_file


def load_checkpoint(path, map_location="cpu"):
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
        raise CheckpointError(f"{path}: expected a dict with 'model' and 'config'")
    return ckpt


def architecture_from_config(cfg):
    """Model sizes and OCR line cap from a checkpoint ``config`` mapping."""
    try:
        return {"d": int(cfg["D"]), "h": int(cfg["H"]), "L": int(cfg["L"]), "ffn_mult": int(cfg["FFN_MULT"]),
                "pdrop": float(cfg["DROP"]), "K": int(cfg["K"])}
    except KeyError as e:
        raise CheckpointError(f"checkpoint config lacks {e}") from None


def temperature_from_checkpoint(ckpt, tau_mode=None, fixed_multiplier=None):
    """Temperature settings: explicit arguments, else ``xstar`` metadata, else the stored parameter."""
    meta = ckpt.get("xstar") or {}
    mode = tau_mode or meta.get("tau_mode") or "stored_parameter"
    mult = fixed_multiplier if fixed_multiplier is not None else meta.get("fixed_multiplier")
    return mode, mult


def build_model(ckpt, tau_mode=None, fixed_multiplier=None, device="cpu"):
    """Instantiate :class:`FusionLateBank` and load the state dict with ``strict=True``.

    Returns ``(model, K)``. The stored ``tau`` and ``null_O`` are loaded as saved.
    """
    arch = architecture_from_config(ckpt["config"])
    mode, mult = temperature_from_checkpoint(ckpt, tau_mode, fixed_multiplier)
    model = FusionLateBank(d=arch["d"], h=arch["h"], L=arch["L"], pdrop=arch["pdrop"], ffn_mult=arch["ffn_mult"],
                           tau_mode=mode, fixed_multiplier=mult)
    check_state_dict_keys(model, ckpt["model"])
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device), arch["K"]


def check_state_dict_keys(model, state):
    expected, found = set(model.state_dict()), set(state)
    if expected != found:
        raise CheckpointError(f"state-dict keys differ: missing {sorted(expected - found)[:5]}, "
                              f"unexpected {sorted(found - expected)[:5]}")
    for k, v in model.state_dict().items():
        if tuple(state[k].shape) != tuple(v.shape):
            raise CheckpointError(f"shape mismatch for {k}: checkpoint {tuple(state[k].shape)} vs model {tuple(v.shape)}")


def checkpoint_identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def save_checkpoint(path, payload):
    """Save to a path that must not exist yet."""
    path = Path(path)
    if path.exists():
        raise CheckpointError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


# ---------------------------------------------------------------------------
# Inference-only exports
# ---------------------------------------------------------------------------

INFERENCE_FORMAT = "xstar-inference-v1"

# Top-level entries of a training checkpoint that inference never reads.
TRAINING_ONLY_KEYS = ("optim", "optimizer", "scheduler", "lr_scheduler", "scaler")
KNOWN_TOP_LEVEL_KEYS = {"model", "config", "epoch", "metrics", "xstar", *TRAINING_ONLY_KEYS}
# Trainer metadata needed only to resume a run exactly.
RESUME_ONLY_META = ("rng", "run_signature")


def is_inference_only(ckpt):
    meta = ckpt.get("xstar") or {}
    return meta.get("format") == INFERENCE_FORMAT or bool(meta.get("inference_only"))


def source_sha256(ckpt):
    """SHA-256 of the training checkpoint an inference export was made from (``None`` otherwise)."""
    return ((ckpt.get("xstar") or {}).get("source") or {}).get("sha256")


def _tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def compare_state_dicts(a, b):
    """Differences between two state dicts (keys, shapes, dtypes, values); empty when exactly equal."""
    problems = []
    if set(a) != set(b):
        problems.append(f"keys differ: only in first {sorted(set(a) - set(b))[:5]}, "
                        f"only in second {sorted(set(b) - set(a))[:5]}")
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            problems.append(f"{k}: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}")
        elif not torch.equal(x, y):
            problems.append(f"{k}: values differ")
    return problems


def resolve_export_temperature(ckpt, tau_mode=None, fixed_multiplier=None):
    """Temperature settings for an export.

    A checkpoint written by this package records its mode; an explicit value must
    agree with it. A checkpoint without that metadata (the original research
    checkpoints) needs ``tau_mode`` stated explicitly, so the export never guesses.
    """
    meta = ckpt.get("xstar") or {}
    if "tau_mode" in meta:
        mode, mult = meta["tau_mode"], meta.get("fixed_multiplier")
        if tau_mode is not None and tau_mode != mode:
            raise CheckpointError(f"source records tau_mode={mode!r}; refusing to export it as {tau_mode!r}")
        if fixed_multiplier is not None and (mult is None or float(fixed_multiplier) != float(mult)):
            raise CheckpointError(f"source records fixed_multiplier={mult!r}; refusing {fixed_multiplier!r}")
        return mode, mult, "source checkpoint metadata"
    if tau_mode is None:
        raise CheckpointError("source checkpoint has no temperature metadata; state --tau-mode "
                              "(stored_parameter for X-STAR fine-tuned checkpoints, fixed with "
                              "--fixed-multiplier 50 for the TextCaps checkpoint)")
    if tau_mode == "fixed" and fixed_multiplier is None:
        raise CheckpointError("--tau-mode fixed needs --fixed-multiplier")
    if tau_mode == "stored_parameter" and fixed_multiplier is not None:
        raise CheckpointError("--fixed-multiplier applies only to --tau-mode fixed")
    return tau_mode, (None if fixed_multiplier is None else float(fixed_multiplier)), "stated at export"


def inference_settings(arch):
    """How the stored weights are used at inference (fixed by this package, recorded for readers)."""
    from xstar.constants import CLIP_MODEL_NAME, TEXT_CONTEXT, VISUAL_TOKENS

    return {
        "model_class": "xstar.fusion.FusionLateBank",
        "clip_model": CLIP_MODEL_NAME,
        "embedding_dim": arch["d"],
        "fusion_blocks": arch["L"],
        "attention_heads": arch["h"],
        "ffn_mult": arch["ffn_mult"],
        "visual_tokens": VISUAL_TOKENS,
        "visual_features": "frozen CLIP ViT-B/32 post-transformer tokens (CLS + 49 patches), projected to 512-d",
        "image_preprocessing": "CLIP ViT-B/32 preprocessing (224x224); see xstar.encoders",
        "ocr_max_lines": arch["K"],
        "ocr_features": "one frozen CLIP text embedding per English OCR line, L2-normalised; up to K lines",
        "ocr_padding": "unused OCR slots replaced by the learned null_O token, then treated as valid",
        "bank_tokens": VISUAL_TOKENS + arch["K"],
        "bank_normalisation": "L2 per token",
        "query_features": f"per-token CLIP text embeddings (context {TEXT_CONTEXT}); valid tokens are token_id != 0",
        "scoring": "ColBERT-style MaxSim: sum over valid query tokens of the max dot product with bank tokens",
        "dropout_at_inference": "disabled (model.eval())",
    }


def export_inference_checkpoint(source, output, experiment, split, tau_mode=None, fixed_multiplier=None):
    """Write an inference-only copy of a training checkpoint and return a summary.

    Model tensors are copied exactly (same keys, shapes, dtypes and values);
    optimiser state and resume-only metadata are dropped. The output must not
    exist. The result loads with :func:`load_checkpoint` and :func:`build_model`.
    """
    from xstar import __version__
    from xstar.utils import git_state, utc_now

    source, output = Path(source), Path(output)
    if output.exists():
        raise CheckpointError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise CheckpointError(f"output directory does not exist: {output.parent}")
    src = load_checkpoint(source)
    if is_inference_only(src):
        raise CheckpointError(f"{source} is already an inference-only export")
    unknown = sorted(set(src) - KNOWN_TOP_LEVEL_KEYS)
    if unknown:
        raise CheckpointError(f"{source} has unrecognised top-level entries {unknown}; refusing to drop them silently")

    arch = architecture_from_config(src["config"])
    mode, mult, mode_origin = resolve_export_temperature(src, tau_mode, fixed_multiplier)
    src_meta = dict(src.get("xstar") or {})
    identity = checkpoint_identity(source)
    removed = {k: {"tensor_bytes": _tensor_bytes(src[k])} for k in TRAINING_ONLY_KEYS if k in src}
    removed.update({f"xstar.{k}": {"tensor_bytes": _tensor_bytes(src_meta[k])} for k in RESUME_ONLY_META
                    if k in src_meta})
    kept_meta = {k: v for k, v in src_meta.items() if k not in RESUME_ONLY_META}

    model_state = {k: v.detach().clone() for k, v in src["model"].items()}
    payload = {
        "model": model_state,
        "config": dict(src["config"]),
        "epoch": src.get("epoch"),
        "metrics": dict(src.get("metrics") or {}),
        "xstar": {
            "format": INFERENCE_FORMAT,
            "inference_only": True,
            "tau_mode": mode,
            "fixed_multiplier": mult,
            "tau_mode_origin": mode_origin,
            "stored_tau": float(model_state["tau"]),
            "ocr_max_lines": arch["K"],
            "inference": inference_settings(arch),
            "source": {"filename": source.name, "sha256": identity["sha256"], "size_bytes": identity["size_bytes"],
                       "experiment": experiment, "split": split, "epoch": src.get("epoch"),
                       "training_metadata": kept_meta},
            "removed_from_source": removed,
            "exporter": {"code_revision": git_state(), "xstar_version": __version__,
                         "torch_version": str(torch.__version__), "created_utc": utc_now()},
        },
    }
    problems = compare_state_dicts(src["model"], payload["model"])
    if problems:
        raise CheckpointError("model tensors changed during export: " + "; ".join(problems[:5]))
    build_model(payload)                                   # strict load before anything is written
    save_checkpoint(output, payload)

    written = load_checkpoint(output)                      # read back with the supported loader
    problems = compare_state_dicts(src["model"], written["model"])
    if problems:
        raise CheckpointError(f"{output} differs from its source after writing: " + "; ".join(problems[:5]))
    out_identity = checkpoint_identity(output)
    return {"source": identity, "output": out_identity, "tau_mode": mode, "fixed_multiplier": mult,
            "stored_tau": payload["xstar"]["stored_tau"], "ocr_max_lines": arch["K"],
            "model_tensors": len(model_state), "model_tensor_bytes": _tensor_bytes(model_state),
            "removed_from_source": removed, "kept_training_metadata": sorted(kept_meta),
            "size_reduction_bytes": identity["size_bytes"] - out_identity["size_bytes"]}
