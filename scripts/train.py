#!/usr/bin/env python3
"""Train CoLITE-CLIP from precomputed frozen CLIP features.

The stage file selects TextCaps pre-training (configs/textcaps_pretrain.yaml)
or X-STAR fine-tuning (configs/xstar_finetune.yaml), including its temperature
behaviour. Every run writes into a new --output-dir: run_metadata.json,
metrics.jsonl and checkpoints/epochNN_*.pt plus checkpoints/best.pt (lowest
validation loss).

Initialisation and resume are different operations:
  --init-checkpoint  load model weights only; new optimiser, schedule and epoch 0.
                     Defaults to training.init_checkpoint of the stage file
                     (a key under 'checkpoints:' in --paths); --no-init disables it.
  --resume           continue a run of this trainer from one of its epoch
                     checkpoints inside --output-dir (optimiser, epoch, step,
                     best value and RNG state are restored). The stage file,
                     data inputs, architecture, temperature, schedule settings
                     and smoke limits must be unchanged, or the resume is
                     refused before anything is written; only --device and
                     --num-workers may differ. To change the experiment,
                     start a new run with --init-checkpoint.

TextCaps pre-training runs on existing TextCaps feature files only; preparing
those features from raw TextCaps data is not implemented here.

Example:
  python scripts/train.py --config configs/xstar_finetune.yaml \\
      --paths configs/paths.yaml --output-dir runs/xstar_finetune --device cuda:0
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import parser, resolve_checkpoint, run  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--config", required=True, help="stage YAML (textcaps_pretrain.yaml or xstar_finetune.yaml)")
    ap.add_argument("--paths", required=True, help="paths YAML (see configs/paths.example.yaml)")
    ap.add_argument("--output-dir", required=True, help="new run directory (must not exist or be empty)")
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    init = ap.add_mutually_exclusive_group()
    init.add_argument("--init-checkpoint", default=None,
                      help="weights to initialise from: a file or a key under 'checkpoints:' in --paths")
    init.add_argument("--no-init", action="store_true", help="train from random initialisation")
    init.add_argument("--resume", default=None, help="epoch checkpoint of this run (inside --output-dir) to resume")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: stage file)")
    ap.add_argument("--max-train-batches", type=int, default=None,
                    help="smoke test: stop each epoch after N batches and stop after one epoch")
    ap.add_argument("--max-val-batches", type=int, default=None, help="smoke test: validate on N batches")
    args = ap.parse_args()

    from xstar.config import PathsConfig, StageConfig
    from xstar.training import train

    stage = StageConfig.load(args.config)
    paths = PathsConfig.load(args.paths)
    init_ckpt = None
    if args.resume is None and not args.no_init:
        key = args.init_checkpoint or stage.training.get("init_checkpoint")
        init_ckpt = resolve_checkpoint(key, paths) if key else None
    result = train(stage, paths, args.output_dir, device=args.device, init_checkpoint=init_ckpt, resume=args.resume,
                   max_train_batches=args.max_train_batches, max_val_batches=args.max_val_batches,
                   num_workers=args.num_workers)
    print(f"Done. Best epoch {result['best_epoch']} (val loss {result['best_val_loss']:.4f}); outputs in {result['output_dir']}")


if __name__ == "__main__":
    run(main)
