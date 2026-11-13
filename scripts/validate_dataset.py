#!/usr/bin/env python3
"""Validate an X-STAR manifest, its splits, images and (optionally) cached features.

Checks, all reported together:
  * manifest: required fields, unique IDs and filenames, image_id = <folder>_<stem>,
    relative '<folder>/<file>' paths, known splits, non-empty query fields;
  * splits: each split list matches the manifest's split exactly, lists are
    disjoint and together cover the manifest;
  * images (with --image-root): every manifest image file exists;
  * features (with --check-features): image-token rows match the split lists;
    packed-query IDs follow the image order; all 13 test query sets have global
    embeddings whose IDs equal the packed IDs and whose cached strings equal
    the manifest fields; the test global image array matches the gallery; every
    image has an OCR file whose E array is [lines, 512], finite, and whose line
    count equals the manifest's normalised OCR lines.

Nothing is modified. Exit status 1 if any check fails.

Example:
  python scripts/validate_dataset.py --paths configs/paths.yaml --check-features
"""

import sys
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xstar.cli import parser, run  # noqa: E402


def main():
    ap = parser(__doc__)
    ap.add_argument("--paths", required=True, help="paths YAML (manifest, split lists, features)")
    ap.add_argument("--image-root", default=None, help="check image presence under this root (default: dataset.root)")
    ap.add_argument("--no-image-check", action="store_true", help="skip the image presence check")
    ap.add_argument("--check-features", action="store_true", help="also validate cached features")
    ap.add_argument("--ocr-variant", default=None, help="OCR variant key (default: features.ocr.variant)")
    ap.add_argument("--ocr-field", default="ocr_lines_en", help="manifest field compared with OCR line counts")
    ap.add_argument("--report", default=None, help="write the findings as JSON to this new file")
    args = ap.parse_args()

    import numpy as np

    from xstar.config import PathsConfig
    from xstar.constants import QUERY_SETS, SPLITS
    from xstar.data import (
        DataError,
        ImageTokens,
        Manifest,
        PackedQueries,
        image_id_from_filename,
        load_global_queries,
        load_image_global,
        load_ocr_embeddings,
        normalize_ocr_lines,
        ocr_path,
        read_lines,
    )
    from xstar.utils import check_not_inside, write_json_new

    paths = PathsConfig.load(args.paths)
    findings = {"errors": [], "checks": {}}

    def check(name, fn):
        try:
            findings["checks"][name] = fn() or "ok"
        except (DataError, ValueError, KeyError, OSError) as e:
            findings["checks"][name] = "FAILED"
            findings["errors"].append(f"{name}: {e}")

    manifest = Manifest.load(paths.manifest)

    def manifest_check():
        problems = manifest.validate(ocr_field=args.ocr_field)
        if problems:
            raise DataError("; ".join(problems))
        return {"records": len(manifest.records), **{s: len(manifest.split_records(s)) for s in SPLITS}}
    check("manifest", manifest_check)

    lists = {}

    def splits_check():
        sets = {}
        for s in SPLITS:
            lst = read_lines(paths.split_list(s))
            lists[s] = lst
            sets[s] = set(lst)
            if len(sets[s]) != len(lst):
                raise DataError(f"{s} list has duplicates")
            if sorted(lst) != manifest.split_filenames(s):
                raise DataError(f"{s} list does not match the manifest's {s} split")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if sets[a] & sets[b]:
                raise DataError(f"{a} and {b} share {len(sets[a] & sets[b])} images")
        if set().union(*sets.values()) != set(manifest.by_filename):
            raise DataError("split lists do not cover the manifest")
        return {s: len(v) for s, v in lists.items()}
    check("splits", splits_check)

    root = Path(args.image_root) if args.image_root else paths.dataset_root
    if not args.no_image_check:
        def images_check():
            if root is None:
                raise DataError("no --image-root and no dataset.root in the paths config")
            absent = [r["filename"] for r in manifest.records if not (root / r["filename"]).is_file()]
            if absent:
                raise DataError(f"{len(absent)} images missing under {root}, e.g. {absent[:5]}")
            return {"present": len(manifest.records)}
        check("images", images_check)

    if args.check_features and lists:
        ids = {s: [image_id_from_filename(f) for f in lists[s]] for s in SPLITS}
        for s in SPLITS:
            check(f"image_tokens.{s}", lambda s=s: {"rows": len(ImageTokens(paths.image_tokens(s), paths.split_list(s)))})
            for qs in (QUERY_SETS if s == "test" else ("english_query",)):
                def packed_check(s=s, qs=qs):
                    pq = PackedQueries(*paths.packed_queries(s, qs))
                    if pq.ids != ids[s]:
                        raise DataError(f"packed IDs differ from the {s} image order")
                    return {"rows": len(pq)}
                check(f"packed_queries.{s}.{qs}", packed_check)
        for qs in QUERY_SETS:
            def global_check(qs=qs):
                vec, id_path, str_path = paths.global_queries("test", qs)
                g, gids = load_global_queries(vec, id_path)
                if gids != PackedQueries(*paths.packed_queries("test", qs)).ids:
                    raise DataError("global-query IDs differ from packed-query IDs")
                strings = read_lines(str_path)
                bad = [i for i, t in zip(gids, strings) if t != manifest.query_text(i, qs)]
                if len(strings) != len(gids) or bad:
                    raise DataError(f"{len(bad)} cached strings differ from the manifest, e.g. {bad[:3]}")
                return {"rows": len(gids)}
            check(f"global_queries.test.{qs}", global_check)
        check("image_global.test", lambda: {"rows": load_image_global(paths.image_global("test"), len(ids["test"])).shape[0]})
        for s in SPLITS:
            def ocr_check(s=s):
                d = paths.ocr_dir(s, args.ocr_variant)
                count_mismatch = []
                for iid in ids[s]:
                    E = load_ocr_embeddings(ocr_path(d, iid))
                    if E.shape[0] != len(normalize_ocr_lines(manifest.record(iid).get(args.ocr_field))):
                        count_mismatch.append(iid)
                    if E.shape[0] and not np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4):
                        raise DataError(f"{iid}: OCR embeddings are not unit-norm")
                if count_mismatch:
                    raise DataError(f"{len(count_mismatch)} OCR files have a line count different from "
                                    f"{args.ocr_field}, e.g. {count_mismatch[:5]}")
                return {"files": len(ids[s]), "directory": str(d)}
            check(f"ocr.{s}", ocr_check)

    for name, result in findings["checks"].items():
        print(f"{'FAIL' if result == 'FAILED' else 'ok  '} {name}")
    for e in findings["errors"]:
        print(f"  - {e}")
    if args.report:
        check_not_inside(args.report, paths.protected_roots())
        write_json_new(args.report, findings)
    print("VALID" if not findings["errors"] else f"INVALID: {len(findings['errors'])} failed checks")
    sys.exit(1 if findings["errors"] else 0)


if __name__ == "__main__":
    run(main)
