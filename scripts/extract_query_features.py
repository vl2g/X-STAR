#!/usr/bin/env python3
"""Extract CLIP ViT-B/32 query features from the supplied English manifest fields.

For each query set (manifest field holding English or back-translated English
text) and the split's image order, writes into a new --output-root, using the
file layout of the original feature cache:

  <packed-dir>/<split>_q_tok[__<set>].npy    float32 [N, 1, 77, 512] per-token
  <packed-dir>/<split>_q_mask[__<set>].npy   bool    [N, 1, 77]  token_id != 0
  <packed-dir>/<split>_q_ids[__<set>].txt    image IDs (the relevant image)
  <english-dir>/<split>_english_query_vitb32.npy, <split>_image_ids.txt,
      <split>_english_query_strings.txt      global English queries [N, 512]
  <to-english-dir>/<split>_<set>_vitb32.npy, _image_ids.txt, _strings.txt
                                             global back-translated queries
  query_features_<split>.json                sets, languages, counts, settings

The "__<set>" suffix is omitted for english_query. Text is whitespace-collapsed
and tokenised with clip.tokenize(truncate=True); an empty query stops the run.
No translation is performed.

Defaults follow the original runs: english_query only for train/val; all 13
sets for test; global embeddings for test only.

Example:
  python scripts/extract_query_features.py --manifest XSTAR_final_manifest2.json \\
      --split test --output-root features/queries
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import add_clip_args, add_limit_arg, add_subset_arg, parser, run  # noqa: E402
from xstar.constants import QUERY_SETS  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--image-list", default=None, help="explicit row order (must match the split's images)")
    ap.add_argument("--output-root", required=True, help="new output directory")
    ap.add_argument("--query-sets", nargs="+", default=None, choices=QUERY_SETS, metavar="SET",
                    help="default: english_query for train/val, all 13 for test")
    ap.add_argument("--global", dest="with_global", choices=("auto", "yes", "no"), default="auto",
                    help="write global embeddings (auto: test split only)")
    ap.add_argument("--packed-dir", default="colbert_packed2")
    ap.add_argument("--english-dir", default="english_query_vitb32_2")
    ap.add_argument("--to-english-dir", default="to_english_vitb32_2")
    ap.add_argument("--batch-size", type=int, default=256)
    add_limit_arg(ap)
    add_subset_arg(ap)
    add_clip_args(ap)
    args = ap.parse_args()

    import numpy as np

    from xstar.constants import QUERY_SET_LANGUAGE
    from xstar.data import Manifest, image_id_from_filename, read_lines, select_ids, split_order
    from xstar.encoders import encode_text_global, encode_text_tokens, load_clip
    from xstar.utils import environment_info, prepare_output_dir, sha256_file, utc_now, write_json_new

    sets = args.query_sets or (list(QUERY_SETS) if args.split == "test" else ["english_query"])
    with_global = args.with_global == "yes" or (args.with_global == "auto" and args.split == "test")
    manifest = Manifest.load(args.manifest)
    order = split_order(manifest, args.split, args.image_list)
    if args.image_ids:
        order = select_ids(order, read_lines(args.image_ids))
    if args.limit is not None:
        order = order[:args.limit]
    ids = [image_id_from_filename(fn) for fn in order]
    texts = {qs: [manifest.query_text(i, qs) for i in ids] for qs in sets}   # raises on empty text

    out = prepare_output_dir(args.output_root, [Path(args.manifest)])
    for d in (args.packed_dir, args.english_dir, args.to_english_dir):
        (out / d).mkdir(exist_ok=True)
    model, _ = load_clip(args.device, args.clip_download_root, args.allow_download)

    record = {}
    for qs in sets:
        suffix = "" if qs == "english_query" else f"__{qs}"
        tok, msk = encode_text_tokens(model, texts[qs], device=args.device, batch_size=args.batch_size)
        pk = out / args.packed_dir
        np.save(pk / f"{args.split}_q_tok{suffix}.npy", tok[:, None])
        np.save(pk / f"{args.split}_q_mask{suffix}.npy", msk[:, None])
        ids_file = pk / f"{args.split}_q_ids{suffix}.txt"
        ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        entry = {"language": QUERY_SET_LANGUAGE[qs], "manifest_field": qs, "rows": len(ids),
                 "ids_sha256": sha256_file(ids_file), "packed_tokens": str(pk / f"{args.split}_q_tok{suffix}.npy")}
        if with_global:
            g = encode_text_global(model, texts[qs], device=args.device, batch_size=args.batch_size)
            if qs == "english_query":
                gd, stem, id_name = out / args.english_dir, f"{args.split}_english_query", f"{args.split}_image_ids.txt"
            else:
                gd, stem, id_name = out / args.to_english_dir, f"{args.split}_{qs}", f"{args.split}_{qs}_image_ids.txt"
            np.save(gd / f"{stem}_vitb32.npy", g)
            (gd / f"{stem}_strings.txt").write_text("\n".join(texts[qs]) + "\n", encoding="utf-8")
            (gd / id_name).write_text("\n".join(ids) + "\n", encoding="utf-8")
            entry["global"] = str(gd / f"{stem}_vitb32.npy")
        record[qs] = entry
        print(f"[{qs}] {len(ids)} queries", flush=True)

    write_json_new(out / f"query_features_{args.split}.json", {
        "created_utc": utc_now(), "command": sys.argv, "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest), "split": args.split, "partial": args.limit is not None or bool(args.image_ids),
        "subset_image_ids_sha256": sha256_file(args.image_ids) if args.image_ids else None,
        "query_sets": record, "text_normalisation": "whitespace collapse; clip.tokenize(truncate=True)",
        "clip_model": "ViT-B/32", "batch_size": args.batch_size, "environment": environment_info()})
    print(f"Wrote {len(sets)} query sets to {out}")


if __name__ == "__main__":
    run(main)
