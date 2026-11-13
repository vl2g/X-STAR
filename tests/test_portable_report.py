"""tools/make_portable_report.py keeps every number and removes machine paths."""

import json
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_portable_report.py"


def test_portable_report_replaces_paths_and_keeps_numbers(tmp_path):
    raw = {"checkpoint": {"path": "/machine/root/runs/x/best.pt", "sha256": "ab"},
           "colite_clip": {"macro_average": {"R@1": 12.3456789}}, "environment": {"platform": "Linux-host", "torch": "2"}}
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps({"device": "cuda:0", "stage_config_file": "/machine/root/c.yaml"}))
    (run / "metrics.jsonl").write_text(json.dumps({"epoch": 1, "val_loss": 0.5}) + "\n")
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps(raw))
    out = tmp_path / "out.json"
    cmd = [sys.executable, str(TOOL), "--eval", str(ev), "--run-dir", str(run), "--replace", "/machine/root=<repo>",
           "--output", str(out)]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    text = out.read_text()
    r = json.loads(text)
    assert "/machine/root" not in text and r["evaluation"]["checkpoint"]["path"] == "<repo>/runs/x/best.pt"
    assert r["evaluation"]["colite_clip"]["macro_average"]["R@1"] == 12.3456789
    assert "platform" not in r["evaluation"]["environment"] and r["training_metrics"][0]["val_loss"] == 0.5
    assert r["sanitization"]["raw_files"]["evaluation"]["name"] == "eval.json"
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0          # never overwrites
