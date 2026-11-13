#!/usr/bin/env python3
"""Rank gallery images for one English query with a CoLITE-CLIP checkpoint.

Query input (one of):
  --query TEXT                  English text, encoded with CLIP ViT-B/32 after
                                whitespace normalisation. Translation from other
                                languages is NOT performed: translate first
                                (X-STAR supplies English back-translations).
  --query-image-id ID --query-set SET
                                a supplied X-STAR query (e.g. hindi_to_english of
                                test image A_image_103), read from the cached
                                packed query features; no CLIP model is needed.

The gallery is the fused token bank of a split (default: all test images). It
is rebuilt from cached features unless --bank-cache points to a bank saved
from the same checkpoint; --save-bank writes one to a new file.

Prints JSON with the encoded query text and the top-K image IDs, relative
image paths and MaxSim scores.

Example:
  python scripts/retrieve.py --paths configs/paths.yaml --checkpoint xstar_main \\
      --query "Lokpriya Gopinath Bordoloi airport exterior entrance" --top-k 5
"""

import json
import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import add_clip_args, fail, parser, resolve_checkpoint, run  # noqa: E402
from xstar.constants import QUERY_SETS  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--paths", required=True, help="paths YAML (see configs/paths.example.yaml)")
    ap.add_argument("--checkpoint", required=True, help="checkpoint file, or a key under 'checkpoints:' in --paths")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"), help="gallery split")
    ap.add_argument("--query", default=None, help="English query text")
    ap.add_argument("--query-image-id", default=None, help="image ID whose supplied query to use")
    ap.add_argument("--query-set", default=None, choices=QUERY_SETS, metavar="SET",
                    help="supplied query set for --query-image-id, e.g. english_query or hindi_to_english")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--bank-cache", default=None, help="load the fused gallery bank from this .npz")
    ap.add_argument("--save-bank", default=None, help="write the fused gallery bank to this new .npz")
    ap.add_argument("--output", default=None, help="write JSON here (new file) instead of stdout")
    add_clip_args(ap)
    args = ap.parse_args()
    if (args.query is None) == (args.query_image_id is None):
        fail("give exactly one of --query or --query-image-id")
    if args.query_image_id is not None and args.query_set is None:
        fail("--query-image-id needs --query-set")

    from xstar.checkpoints import build_model, load_checkpoint
    from xstar.config import PathsConfig
    from xstar.data import ImageTokens, Manifest, OcrBank, PackedQueries, sanitize
    from xstar.retrieval import build_bank, load_bank, rank, save_bank
    from xstar.utils import check_not_inside, sha256_file, write_json_new

    paths = PathsConfig.load(args.paths)
    ckpt_path = resolve_checkpoint(args.checkpoint, paths)
    ckpt_sha = sha256_file(ckpt_path)
    model, K = build_model(load_checkpoint(ckpt_path), device=args.device)
    images = ImageTokens(paths.image_tokens(args.split), paths.split_list(args.split))
    manifest = Manifest.load(paths.manifest)

    bank_details = {"split": args.split, "K": K, "ocr_dir": str(paths.ocr_dir(args.split))}
    if args.bank_cache:
        X, X_mask, gallery_ids = load_bank(args.bank_cache, ckpt_sha, expected_ids=images.ids,
                                           expected_details=bank_details)
    else:
        ocr = OcrBank(paths.ocr_dir(args.split), K)
        ocr.check_ids(images.ids)
        X, X_mask = build_bank(model, images, images.ids, ocr, device=args.device)
        gallery_ids = images.ids
        if args.save_bank:
            check_not_inside(args.save_bank, paths.protected_roots())
            save_bank(args.save_bank, X, X_mask, gallery_ids, ckpt_sha, bank_details)

    if args.query is not None:
        from xstar.encoders import encode_text_tokens, load_clip

        text = sanitize(args.query)
        if not text:
            fail("empty query")
        clip_model, _ = load_clip(args.device, args.clip_download_root, args.allow_download)
        Q, Qm = encode_text_tokens(clip_model, [text], device=args.device)
        source = "encoded_with_clip"
    else:
        packed = PackedQueries(*paths.packed_queries(args.split, args.query_set))
        if args.query_image_id not in packed.ids:
            fail(f"{args.query_image_id} has no {args.query_set} query in the {args.split} split")
        row = packed.ids.index(args.query_image_id)
        Q, Qm = packed.single_caption(slice(row, row + 1))
        text = manifest.query_text(args.query_image_id, args.query_set)
        source = f"cached_packed_features:{args.query_set}"

    top, scores = rank(Q, Qm, X, X_mask, top_k=args.top_k, device=args.device)
    results = []
    for r, (j, s) in enumerate(zip(top[0].tolist(), scores[0].tolist()), start=1):
        rec = manifest.record(gallery_ids[j])
        results.append({"rank": r, "image_id": gallery_ids[j], "filename": rec["filename"], "score": s})
    payload = {"query_text_encoded": text, "query_source": source, "split": args.split,
               "gallery_size": len(gallery_ids), "checkpoint_sha256": ckpt_sha, "results": results}
    if args.query_image_id is not None:
        payload["relevant_image_id"] = args.query_image_id
    if args.output:
        check_not_inside(args.output, paths.protected_roots())
        write_json_new(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run(main)
