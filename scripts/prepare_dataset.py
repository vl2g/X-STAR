#!/usr/bin/env python3
"""Assemble the X-STAR image layout from separately obtained source datasets.

X-STAR does not redistribute images. Obtain them from their providers under
their terms, then run this script to place them under --output-root as
<folder>/<file>.jpg, exactly as listed in the manifest:

  * folders A-G, I-L: Bharat Scene Text Dataset detection images; the X-STAR
    filename equals the path inside the BSTD detection folder (--bstd-root);
  * folder N: RCTW-17 training images, mapped by dataset/rctw17_chinese_mapping.csv
    (N/image_<k>.jpg <- raw/train_images/image_<k>.jpg under --rctw-root) and
    verified by SHA-256.

The split is never re-drawn: images are taken from the manifest's fixed
records (optionally only some --splits). Nothing is downloaded. Existing files
under --output-root are never overwritten: an identical file is accepted, a
different one is an error. Source datasets are only read.

Modes:
  --dry-run      report what would be done; write nothing
  --verify-only  check an existing --output-root against the sources/checksums
  (default)      copy (or --link symlink/hardlink) missing images, then verify

Example:
  python scripts/prepare_dataset.py --manifest XSTAR_final_manifest2.json \\
      --bstd-root /path/to/BharatSceneText/detection --rctw-root /path/to/RCTW-17 \\
      --output-root /path/to/XSTAR_dataset --dry-run
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import add_limit_arg, parser, run  # noqa: E402

DEFAULT_MAPPING = Path(__file__).resolve().parents[1] / "dataset" / "rctw17_chinese_mapping.csv"


def main():
    ap = parser(__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True, help="X-STAR image root to populate")
    ap.add_argument("--bstd-root", default=None, help="BSTD detection folder containing A/, B/, ...")
    ap.add_argument("--rctw-root", default=None, help="RCTW-17 root containing raw/train_images/")
    ap.add_argument("--chinese-mapping", default=str(DEFAULT_MAPPING), help="CSV mapping for folder N")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=("train", "val", "test"))
    ap.add_argument("--link", choices=("copy", "symlink", "hardlink"), default="copy")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    ap.add_argument("--report", default=None, help="write a JSON report to this new file")
    add_limit_arg(ap)
    args = ap.parse_args()

    import csv
    import filecmp
    import os
    import shutil

    from xstar.constants import CHINESE_FOLDER
    from xstar.data import DataError, Manifest
    from xstar.utils import check_not_inside, sha256_file, utc_now, write_json_new

    manifest = Manifest.load(args.manifest)
    records = sorted((r for r in manifest.records if r["split"] in args.splits), key=lambda r: r["filename"])
    if args.limit is not None:
        records = records[:args.limit]
    with open(args.chinese_mapping, "r", encoding="utf-8") as f:
        mapping = {row["xstar_filename"]: row for row in csv.DictReader(f)}
    all_n = {r["filename"] for r in manifest.records if r["filename"].startswith(CHINESE_FOLDER + "/")}
    if set(mapping) != all_n:
        raise DataError(f"{args.chinese_mapping} does not cover exactly the manifest's {len(all_n)} Chinese images")

    out = Path(args.output_root)
    bstd = Path(args.bstd_root) if args.bstd_root else None
    rctw = Path(args.rctw_root) if args.rctw_root else None
    if not args.dry_run and not args.verify_only:
        check_not_inside(out, [p for p in (bstd, rctw) if p])

    plan, problems = [], []
    for r in records:
        fn = r["filename"]
        if fn.startswith(CHINESE_FOLDER + "/"):
            if rctw is None:
                problems.append(f"{fn}: --rctw-root not given")
                continue
            src, sha = rctw / mapping[fn]["rctw17_relative_path"], mapping[fn]["sha256"]
        else:
            if bstd is None:
                problems.append(f"{fn}: --bstd-root not given")
                continue
            src, sha = bstd / fn, None
        plan.append((fn, src, sha))

    counts = {"planned": len(plan), "source_sha256_ok": 0, "placed": 0, "already_present": 0, "placed_sha256_ok": 0}
    for fn, src, sha in plan:
        dst = out / fn
        if not src.is_file():
            problems.append(f"{fn}: source missing at {src}")
            continue
        if sha is not None and (args.dry_run or args.verify_only or not dst.exists()):
            if sha256_file(src) != sha:
                problems.append(f"{fn}: source checksum differs from the mapping")
                continue
            counts["source_sha256_ok"] += 1
        if args.dry_run:
            continue
        if dst.exists() or dst.is_symlink():
            if not filecmp.cmp(src, dst, shallow=False):
                problems.append(f"{fn}: existing file differs from its source; not overwritten")
                continue
            counts["already_present"] += 1
        elif args.verify_only:
            problems.append(f"{fn}: missing from --output-root")
            continue
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.link == "symlink":
                os.symlink(src.resolve(), dst)
            elif args.link == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            counts["placed"] += 1
        if sha is not None:
            if sha256_file(dst) != sha:
                problems.append(f"{fn}: checksum mismatch after placement")
                continue
            counts["placed_sha256_ok"] += 1

    mode_name = "dry-run" if args.dry_run else "verify-only" if args.verify_only else args.link
    print(f"mode={mode_name} splits={args.splits} " + " ".join(f"{k}={v}" for k, v in counts.items()))
    for p in problems[:20]:
        print(f"  - {p}")
    if len(problems) > 20:
        print(f"  ... {len(problems) - 20} more")
    if args.report:
        check_not_inside(args.report, [p for p in (bstd, rctw) if p])
        write_json_new(args.report, {"created_utc": utc_now(), "command": sys.argv, "mode": mode_name,
                                     "partial": args.limit is not None, "counts": counts, "problems": problems})
    print("OK" if not problems else f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    run(main)
