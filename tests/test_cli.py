"""Command-line scripts: --help works and loads neither CLIP nor model weights."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))

PROBE = """
import runpy, sys
sys.argv = [{path!r}, "--help"]
try:
    runpy.run_path({path!r}, run_name="__main__")
except SystemExit as e:
    assert e.code == 0, e.code
print("clip" in sys.modules, "torch" in sys.modules)
"""


def test_all_planned_scripts_exist():
    names = {p.stem for p in SCRIPTS}
    assert names == {"validate_dataset", "prepare_dataset", "extract_image_features", "extract_query_features",
                     "extract_ocr_features", "train", "evaluate", "retrieve", "export_inference_checkpoint"}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_help_is_light(script):
    out = subprocess.run([sys.executable, "-c", PROBE.format(path=str(script))], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[-1] == "False False", f"--help imported heavy modules: {lines[-1]}"
    assert "usage:" in out.stdout


def test_readable_error_for_missing_config(tmp_path):
    script = next(p for p in SCRIPTS if p.stem == "evaluate")
    out = subprocess.run([sys.executable, str(script), "--paths", str(tmp_path / "nope.yaml"), "--checkpoint", "x",
                          "--output-dir", str(tmp_path / "o"), "--experiment-name", "e"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 2 and out.stderr.startswith("error:") and "Traceback" not in out.stderr
