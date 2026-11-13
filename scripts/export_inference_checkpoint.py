#!/usr/bin/env python3
"""Export an inference-only copy of a CoLITE-CLIP training checkpoint.

The model tensors are copied exactly (same keys, shapes, dtypes and values; no
FP16 conversion, quantisation or pruning). Optimiser state and resume-only
metadata (RNG state, run signature) are dropped. The output keeps the
architecture config, epoch and metrics, and records the temperature mode,
stored tau, OCR line cap, inference settings, the source checkpoint's SHA-256,
experiment and split, and the exporter's code revision. It loads with the
usual loaders (evaluate.py, retrieve.py) but cannot resume training.

The output file must not exist. Checkpoints without temperature metadata (the
original research checkpoints) need --tau-mode: stored_parameter for X-STAR
fine-tuned checkpoints; fixed with --fixed-multiplier 50 for TextCaps.

Example:
  python scripts/export_inference_checkpoint.py \\
      --input /path/to/checkpoints_colbert_XSTAR_5/best.pt --tau-mode stored_parameter \\
      --experiment xstar_main_original --split original \\
      --output runs/inference_checkpoints/xstar_main_original__inference.pt \\
      --report runs/inference_checkpoints/xstar_main_original__export.json
"""

import json
import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import parser, run  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--input", required=True, help="training checkpoint to export (read only)")
    ap.add_argument("--output", required=True, help="new inference-only checkpoint file (must not exist)")
    ap.add_argument("--experiment", required=True, help="experiment name recorded as provenance")
    ap.add_argument("--split", required=True, help="dataset split definition the model was trained on, "
                    "e.g. original")
    ap.add_argument("--tau-mode", default=None, choices=("stored_parameter", "fixed"),
                    help="required when the source has no temperature metadata; must match it otherwise")
    ap.add_argument("--fixed-multiplier", type=float, default=None, help="multiplier for --tau-mode fixed")
    ap.add_argument("--report", default=None, help="also write the export summary to this new JSON file")
    args = ap.parse_args()

    from xstar.checkpoints import export_inference_checkpoint
    from xstar.utils import write_json_new

    if args.report and Path(args.report).exists():
        raise FileExistsError(f"refusing to overwrite {args.report}")
    summary = export_inference_checkpoint(args.input, args.output, experiment=args.experiment, split=args.split,
                                          tau_mode=args.tau_mode, fixed_multiplier=args.fixed_multiplier)
    if args.report:
        write_json_new(args.report, summary)
    src, out = summary["source"], summary["output"]
    print(json.dumps(summary, indent=2))
    print(f"Exported {out['size_bytes']:,} bytes from {src['size_bytes']:,} "
          f"({100 * out['size_bytes'] / src['size_bytes']:.1f}%); model tensors identical to the source")


if __name__ == "__main__":
    run(main)
