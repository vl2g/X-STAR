"""Hashing, output-location safety and provenance helpers."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from xstar.errors import OutputSafetyError


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolved(path):
    return Path(os.path.realpath(os.path.expanduser(str(path))))


def check_not_inside(output, protected):
    """Refuse an output path that equals, contains or lies inside a protected input.

    ``protected`` may hold directories (dataset root, feature cache, checkpoint
    directories) and files. Symlinks are resolved before comparing.
    """
    out = _resolved(output)
    for item in protected:
        if item is None:
            continue
        p = _resolved(item)
        if out == p or p in out.parents or out in p.parents:
            raise OutputSafetyError(
                f"output location {out} overlaps protected input {p}; choose a new experiment directory")


def prepare_output_dir(path, protected=(), allow_existing=False):
    """Create a new output directory after the safety checks.

    An existing non-empty directory is refused unless ``allow_existing`` is set
    (used only when genuinely resuming a run in its own directory).
    """
    out = _resolved(path)
    check_not_inside(out, protected)
    if out.exists() and any(out.iterdir()) and not allow_existing:
        raise OutputSafetyError(f"output directory {out} already exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json_new(path, payload):
    """Write JSON to a file that must not exist yet."""
    path = Path(path)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def git_state(repo_dir=None):
    repo_dir = repo_dir or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_info():
    import numpy as np
    import torch

    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "executable": sys.executable}
    try:
        from importlib.metadata import version
        info["clip"] = version("clip")
    except Exception:
        info["clip"] = None
    from xstar import __version__
    info["xstar"] = __version__
    info["git"] = git_state()
    return info
