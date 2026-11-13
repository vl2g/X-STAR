#!/usr/bin/env python3
"""Make a portable copy of an evaluation result (and optional training-run metadata).

Replaces machine-specific path prefixes with logical labels (--replace
PREFIX=LABEL, longest prefix first), drops the host platform string, and checks
that every number and boolean in the copy equals the raw input. The raw files
are not modified; their SHA-256 values are recorded in a `sanitization` block.

Example:
  python tools/make_portable_report.py --eval runs/.../X__complete__eval.json \\
      --run-dir runs/xstar_finetune \\
      --replace /path/to/X-STAR=<repo> --output docs/evaluation/result.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def numbers(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, f"{path}[{i}]")
    elif isinstance(x, (int, float, bool)) or x is None:
        yield path, x


def clean(x, rules):
    if isinstance(x, dict):
        return {k: clean(v, rules) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v, rules) for v in x]
    if isinstance(x, str):
        for src, dst in rules:
            x = x.replace(src, dst)
    return x


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="raw evaluation JSON from scripts/evaluate.py")
    ap.add_argument("--run-dir", default=None, help="training run directory (run_metadata.json, metrics.jsonl)")
    ap.add_argument("--replace", action="append", default=[], metavar="PREFIX=LABEL")
    ap.add_argument("--output", required=True, help="new portable JSON file")
    args = ap.parse_args()

    rules = sorted((tuple(r.split("=", 1)) for r in args.replace), key=lambda r: -len(r[0]))
    raw = {"evaluation": json.loads(Path(args.eval).read_text(encoding="utf-8"))}
    files = {"evaluation": args.eval}
    if args.run_dir:
        run = Path(args.run_dir)
        raw["training_run"] = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        raw["training_metrics"] = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
        files["run_metadata"] = run / "run_metadata.json"
        files["metrics"] = run / "metrics.jsonl"
    out = clean(raw, rules)
    for part in out.values():
        if isinstance(part, dict) and isinstance(part.get("environment"), dict):
            part["environment"].pop("platform", None)
    raw_nums = {k: v for k, v in numbers(raw) if ".environment.platform" not in k}
    if dict(numbers(out)) != raw_nums:
        sys.exit("error: numeric content changed during sanitization")
    out["sanitization"] = {
        "raw_files": {k: {"name": Path(v).name, "sha256": sha(v)} for k, v in files.items()},
        "path_prefixes_replaced_by": [label for _, label in rules],
        "environment_platform_removed": True,
        "numeric_values_identical_to_raw": True,
    }
    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    leaked = [src for src, _ in rules if src in text]
    if leaked:
        sys.exit(f"error: unreplaced prefixes remain: {leaked}")
    with open(args.output, "x", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
