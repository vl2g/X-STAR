#!/usr/bin/env python3
"""Evaluate a CoLITE-CLIP checkpoint and the CLIP global baseline on X-STAR.

Scores every query set against the full split gallery (1,289 test images),
resolves relevance by image ID, and writes one uniquely named JSON result file
into a new experiment directory. Runs on the CPU unless --device names a GPU.

A complete benchmark run needs all 13 query sets and the global CLIP query and
image features. The gallery must contain exactly the manifest's split images,
and every query set exactly one query per gallery image, with packed and global
query IDs aligned; anything else stops the run before the model is loaded.
--query-sets, --max-queries and --allow-incomplete-queries produce a result
explicitly marked partial (always against the full gallery).

Example:
  python scripts/evaluate.py --paths configs/paths.yaml --checkpoint xstar_main \\
      --output-dir runs/eval_main --experiment-name main_checkpoint
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import parser, resolve_checkpoint, run  # noqa: E402
from xstar.constants import QUERY_SETS  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--paths", required=True, help="paths YAML (see configs/paths.example.yaml)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint file, or a key under 'checkpoints:' in --paths")
    ap.add_argument("--output-dir", required=True, help="new experiment directory for the result file")
    ap.add_argument("--experiment-name", required=True, help="name used in the result file name")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--query-sets", nargs="+", default=list(QUERY_SETS), choices=QUERY_SETS, metavar="SET",
                    help="query sets to evaluate (default: all 13; fewer marks the result partial)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="evaluate only the first N queries of each set against the full gallery (partial)")
    ap.add_argument("--allow-incomplete-queries", action="store_true",
                    help="accept query caches that cover only part of the split (result marked partial); "
                         "duplicate or unknown query IDs are always errors")
    ap.add_argument("--ocr-variant", default=None, help="OCR variant key from --paths (default: features.ocr.variant)")
    ap.add_argument("--null-visual", action="store_true",
                    help="ablation: feed zero visual tokens to the fusion model (main-result notebook cell 14)")
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ap.add_argument("--bank-batch", type=int, default=256, help="images per fusion batch")
    ap.add_argument("--query-chunk", type=int, default=32, help="queries per MaxSim chunk")
    ap.add_argument("--gallery-chunk", type=int, default=128, help="gallery images per MaxSim chunk")
    args = ap.parse_args()

    from xstar.config import PathsConfig
    from xstar.evaluation import evaluate
    from xstar.utils import environment_info, prepare_output_dir, utc_now, write_json_new

    paths = PathsConfig.load(args.paths)
    ckpt = resolve_checkpoint(args.checkpoint, paths)
    out = prepare_output_dir(args.output_dir, paths.protected_roots(), allow_existing=True)
    result = evaluate(paths, ckpt, split=args.split, query_sets=args.query_sets, ocr_variant=args.ocr_variant,
                      device=args.device, bank_batch=args.bank_batch, query_chunk=args.query_chunk,
                      gallery_chunk=args.gallery_chunk, max_queries=args.max_queries, null_visual=args.null_visual,
                      allow_incomplete_queries=args.allow_incomplete_queries)
    tag = "complete" if result["complete_benchmark"] else "partial"
    ablation = "__null_visual" if args.null_visual else ""
    name = (f"{args.experiment_name}__ckpt-{result['checkpoint']['sha256'][:12]}__{args.split}"
            f"{ablation}__{tag}__eval.json")
    payload = {"experiment_name": args.experiment_name, "created_utc": utc_now(), "command": sys.argv,
               "paths_config": str(Path(args.paths).resolve()), **result, "environment": environment_info()}
    path = write_json_new(out / name, payload)
    m, b = result["colite_clip"]["macro_average"], result["clip_baseline"]["macro_average"]
    print(f"Macro average ({tag}): CoLITE-CLIP {m['R@1']:.2f}/{m['R@5']:.2f}/{m['R@10']:.2f} | "
          f"CLIP {b['R@1']:.2f}/{b['R@5']:.2f}/{b['R@10']:.2f}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    run(main)
