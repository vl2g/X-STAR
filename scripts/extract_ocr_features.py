#!/usr/bin/env python3
"""Encode supplied English OCR lines with the CLIP ViT-B/32 text encoder.

For each image of the split, the OCR lines in --ocr-field are whitespace-
collapsed, empty lines are dropped, and each remaining complete line is encoded
independently (batching is only for speed) into one L2-normalised 512-d
vector. No OCR recognition or translation is run.

Writes one pickle-free file per image into a new --output-dir:
  <image_id>.npz   E      float32 [L, 512]  line embeddings, in line order
                   lines  Unicode [L]       the encoded strings
                   format "xstar-ocr-lines-v1"
Images without OCR lines get E of shape [0, 512] (the model then fills every
OCR slot with its learned null token). A summary goes to ocr_extraction.json.

OCR variants are kept apart by field and output directory:
  --ocr-field ocr_lines_en   recognised OCR (primary manifest; default)
  --ocr-field gt_lines_en    ground-truth OCR (GT-OCR manifest variant)

Example:
  python scripts/extract_ocr_features.py --manifest XSTAR_final_manifest2.json \\
      --split test --output-dir features/ocr_test_recognized
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import add_clip_args, add_limit_arg, parser, run  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--image-list", default=None, help="explicit image order (must match the split's images)")
    ap.add_argument("--ocr-field", default="ocr_lines_en", help="manifest field with English OCR lines")
    ap.add_argument("--output-dir", required=True, help="new output directory")
    ap.add_argument("--batch-size", type=int, default=256, help="lines per encoder batch within one image")
    add_limit_arg(ap)
    add_clip_args(ap)
    args = ap.parse_args()

    from xstar.constants import OCR_FORMAT
    from xstar.data import DataError, Manifest, image_id_from_filename, normalize_ocr_lines, split_order, write_ocr_file
    from xstar.encoders import encode_text_global, load_clip
    from xstar.utils import environment_info, prepare_output_dir, sha256_file, utc_now, write_json_new

    manifest = Manifest.load(args.manifest)
    order = split_order(manifest, args.split, args.image_list)
    if args.limit is not None:
        order = order[:args.limit]
    ids = [image_id_from_filename(fn) for fn in order]
    bad = [i for i in ids if not isinstance(manifest.record(i).get(args.ocr_field), list)]
    if bad:
        raise DataError(f"{len(bad)} records lack a list field {args.ocr_field!r}, e.g. {bad[:5]}")

    out = prepare_output_dir(args.output_dir, [Path(args.manifest)])
    model, _ = load_clip(args.device, args.clip_download_root, args.allow_download)
    empty, lines_total = 0, 0
    for n, iid in enumerate(ids, start=1):
        lines = normalize_ocr_lines(manifest.record(iid)[args.ocr_field])
        E = encode_text_global(model, lines, device=args.device, batch_size=args.batch_size)
        write_ocr_file(out / f"{iid}.npz", E, lines)
        empty += not lines
        lines_total += len(lines)
        if n % 500 == 0:
            print(f"{n}/{len(ids)} images", flush=True)

    write_json_new(out / "ocr_extraction.json", {
        "created_utc": utc_now(), "command": sys.argv, "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest), "split": args.split, "ocr_field": args.ocr_field,
        "format": OCR_FORMAT, "images": len(ids), "images_without_lines": empty, "lines": lines_total,
        "partial": args.limit is not None, "normalisation": "whitespace collapse; empty lines dropped",
        "clip_model": "ViT-B/32", "environment": environment_info()})
    print(f"Wrote {len(ids)} OCR files to {out} ({empty} without lines)")


if __name__ == "__main__":
    run(main)
