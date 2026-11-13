#!/usr/bin/env python3
"""Extract frozen CLIP ViT-B/32 image features for one X-STAR split.

Writes into a new --output-dir:
  <split>_image_tokens_postxfmr_vitb32.npy   float32 [N, 50, 512]  post-transformer
                                              projected tokens (CLS + 49 patches)
  <split>_image_global_embeddings.npy        float32 [N, 512] L2-normalised global
                                              embeddings (only with --global)
  image_filenames_<split>.txt                row order of both arrays
  <split>_image_extraction.json              settings, counts, skipped images

Image order is the sorted filenames of the split (the released split lists),
or --image-list if given (it must contain exactly the split's images).
Preprocessing follows the original extractor: EXIF transpose, RGB, clamp of
images above 36 MP or 6000 px, CLIP's 224x224 preprocess.

By default any missing or unreadable image stops the run. --skip-unreadable
drops such images and records them; the output is then not the benchmark
split and is marked partial. Truncated JPEG data is tolerated as in the
original extractor unless --strict-images is given.

Example:
  python scripts/extract_image_features.py --manifest XSTAR_final_manifest2.json \\
      --image-root /path/to/XSTAR_dataset --split test --global --output-dir features/test_images
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import add_clip_args, add_limit_arg, add_subset_arg, parser, run  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True, help="folder holding <folder>/<file>.jpg images")
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--image-list", default=None, help="explicit row order (must match the split's images)")
    ap.add_argument("--output-dir", required=True, help="new output directory")
    ap.add_argument("--global", dest="with_global", action="store_true", help="also write global image embeddings")
    ap.add_argument("--batch-size", type=int, default=1, help="images per forward pass (original: 1)")
    ap.add_argument("--skip-unreadable", action="store_true",
                    help="skip missing/unreadable images instead of failing (output marked partial)")
    ap.add_argument("--strict-images", action="store_true", help="do not tolerate truncated image files")
    add_limit_arg(ap)
    add_subset_arg(ap)
    add_clip_args(ap)
    args = ap.parse_args()

    import numpy as np
    import torch

    from xstar.data import DataError, Manifest, read_lines, select_ids, split_order
    from xstar.encoders import configure_pillow, encode_image_global, encode_image_tokens, load_clip, load_image
    from xstar.utils import environment_info, prepare_output_dir, sha256_file, utc_now, write_json_new

    manifest = Manifest.load(args.manifest)
    order = split_order(manifest, args.split, args.image_list)
    if args.image_ids:
        order = select_ids(order, read_lines(args.image_ids))
    if args.limit is not None:
        order = order[:args.limit]
    image_root = Path(args.image_root)
    out = prepare_output_dir(args.output_dir, [image_root, Path(args.manifest)])
    configure_pillow(tolerate_truncated=not args.strict_images)
    model, preprocess = load_clip(args.device, args.clip_download_root, args.allow_download)

    tokens = np.lib.format.open_memmap(out / f".{args.split}_tokens.partial.npy", mode="w+", dtype=np.float32,
                                       shape=(len(order), 50, 512))
    glob = np.zeros((len(order), 512), dtype=np.float32) if args.with_global else None
    kept, skipped = [], []

    def flush(batch_files, batch_tensors):
        x = torch.stack(batch_tensors).to(args.device)
        t = encode_image_tokens(model, x).float().cpu().numpy()
        g = encode_image_global(model, x).float().cpu().numpy() if args.with_global else None
        for i, fn in enumerate(batch_files):
            if not np.isfinite(t[i]).all() or (g is not None and not np.isfinite(g[i]).all()):
                raise DataError(f"non-finite features for {fn}")
            tokens[len(kept)] = t[i]
            if g is not None:
                glob[len(kept)] = g[i]
            kept.append(fn)

    batch_files, batch_tensors = [], []
    for n, fn in enumerate(order, start=1):
        try:
            img = preprocess(load_image(image_root / fn))
        except Exception as e:  # missing, unreadable or not an image
            if not args.skip_unreadable:
                raise DataError(f"cannot read {image_root / fn}: {e}") from None
            skipped.append({"filename": fn, "reason": f"{type(e).__name__}: {e}"})
            continue
        batch_files.append(fn)
        batch_tensors.append(img)
        if len(batch_files) == args.batch_size:
            flush(batch_files, batch_tensors)
            batch_files, batch_tensors = [], []
        if n % 200 == 0:
            print(f"{n}/{len(order)} images", flush=True)
    if batch_files:
        flush(batch_files, batch_tensors)

    tok_name = f"{args.split}_image_tokens_postxfmr_vitb32.npy"
    np.save(out / tok_name, np.asarray(tokens[:len(kept)]))
    (out / f".{args.split}_tokens.partial.npy").unlink()
    if glob is not None:
        np.save(out / f"{args.split}_image_global_embeddings.npy", glob[:len(kept)])
    (out / f"image_filenames_{args.split}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    partial = bool(skipped) or args.limit is not None or bool(args.image_ids)
    write_json_new(out / f"{args.split}_image_extraction.json", {
        "created_utc": utc_now(), "command": sys.argv, "manifest": str(Path(args.manifest).resolve()),
        "split": args.split, "requested": len(order), "written": len(kept), "skipped": skipped,
        "partial": partial, "subset_image_ids_sha256": sha256_file(args.image_ids) if args.image_ids else None,
        "batch_size": args.batch_size, "tolerate_truncated": not args.strict_images,
        "global_embeddings": args.with_global, "clip_model": "ViT-B/32", "environment": environment_info()})
    print(f"Wrote {len(kept)} images to {out}" + (" (partial)" if partial else ""))


if __name__ == "__main__":
    run(main)
