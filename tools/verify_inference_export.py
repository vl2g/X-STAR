#!/usr/bin/env python3
"""Check that an inference-only export behaves exactly like its source checkpoint on real data.

Compares, for one source/export pair:
  1. model tensors (keys, shapes, dtypes, values), config, epoch and metrics;
  2. temperature mode, multiplier and stored tau, and the OCR line cap K, after
     loading both with the supported loader (strict model initialisation);
  3. the fused gallery bank and mask for the complete split gallery;
  4. MaxSim scores of every query in the chosen cached query sets against the
     complete gallery, and the stable-sorted rankings (gallery-order ties);
  5. CoLITE-CLIP Recall@1/5/10 recomputed from both files with the evaluator's
     own functions (a consistency check, not a new result);
  6. optionally, recorded top-5 rankings (``--expected-rankings``, a JSON with
     ``examples`` holding ``image_id``, ``query_set`` and ``top5``).

Settings match scripts/evaluate.py: float32, TF32 disabled, same chunk sizes.
Writes a JSON report to a new file; exit status 1 if anything differs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import xstar  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True, help="paths YAML for the split the model belongs to")
    ap.add_argument("--source", required=True, help="training checkpoint")
    ap.add_argument("--export", required=True, help="inference-only export of --source")
    ap.add_argument("--split", default="test")
    ap.add_argument("--query-sets", nargs="+", default=None, help="cached query sets (default: all 13)")
    ap.add_argument("--expected-rankings", default=None, help="JSON with recorded top-5 rankings to re-check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bank-batch", type=int, default=256)
    ap.add_argument("--query-chunk", type=int, default=32)
    ap.add_argument("--gallery-chunk", type=int, default=128)
    ap.add_argument("--output", required=True, help="new JSON report file")
    args = ap.parse_args()

    import torch

    from xstar.checkpoints import build_model, checkpoint_identity, compare_state_dicts, load_checkpoint
    from xstar.config import PathsConfig
    from xstar.constants import QUERY_SETS
    from xstar.data import ImageTokens, OcrBank, PackedQueries
    from xstar.evaluation import KS, TIE_POLICY, macro_average, recall_at_k, relevance_indices, relevant_ranks
    from xstar.retrieval import build_bank
    from xstar.scoring import maxsim_scores
    from xstar.utils import environment_info, sha256_file, utc_now, write_json_new

    if Path(args.output).exists():
        sys.exit(f"error: refusing to overwrite {args.output}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = args.device
    query_sets = args.query_sets or list(QUERY_SETS)
    paths = PathsConfig.load(args.paths)
    failures, report = [], {"created_utc": utc_now(), "device": dev}
    if dev.startswith("cuda"):
        report["device_name"] = torch.cuda.get_device_name(torch.device(dev))

    src, exp = load_checkpoint(args.source), load_checkpoint(args.export)
    tensor_diffs = compare_state_dicts(src["model"], exp["model"])
    report["checkpoints"] = {"source": checkpoint_identity(args.source), "export": checkpoint_identity(args.export)}
    meta = exp.get("xstar") or {}
    if (meta.get("source") or {}).get("sha256") != report["checkpoints"]["source"]["sha256"]:
        failures.append("export does not record this source's SHA-256")
    report["tensors"] = {"count": len(src["model"]), "identical": not tensor_diffs, "differences": tensor_diffs[:20],
                         "dtypes": sorted({str(t.dtype) for t in exp["model"].values()})}
    same_meta = {k: src.get(k) == exp.get(k) for k in ("config", "epoch", "metrics")}
    report["metadata_equal"] = same_meta
    failures += [f"tensor mismatch: {d}" for d in tensor_diffs[:5]]
    failures += [f"{k} differs" for k, ok in same_meta.items() if not ok]

    m_src, K_src = build_model(src, device=dev)
    m_exp, K_exp = build_model(exp, device=dev)
    temp = {"source": {"tau_mode": m_src.tau_mode, "fixed_multiplier": m_src.fixed_multiplier,
                       "stored_tau": float(m_src.tau), "temperature": float(m_src.temperature()), "K": K_src},
            "export": {"tau_mode": m_exp.tau_mode, "fixed_multiplier": m_exp.fixed_multiplier,
                       "stored_tau": float(m_exp.tau), "temperature": float(m_exp.temperature()), "K": K_exp}}
    temp["identical"] = temp["source"] == temp["export"]
    report["temperature_and_K"] = temp
    if not temp["identical"]:
        failures.append("temperature or K differs")

    images = ImageTokens(paths.image_tokens(args.split), paths.split_list(args.split))
    ocr = OcrBank(paths.ocr_dir(args.split), K_src)
    ocr.check_ids(images.ids)
    gallery = images.ids
    t0 = time.time()
    Xa, Ma = build_bank(m_src, images, gallery, ocr, batch_size=args.bank_batch, device=dev)
    Xb, Mb = build_bank(m_exp, images, gallery, ocr, batch_size=args.bank_batch, device=dev)
    report["gallery"] = {"split": args.split, "size": len(gallery), "manifest_sha256": sha256_file(paths.manifest),
                         "split_list_sha256": sha256_file(paths.split_list(args.split)),
                         "bank_shape": list(Xa.shape), "bank_identical": bool(torch.equal(Xa, Xb)),
                         "bank_max_abs_diff": float((Xa - Xb).abs().max()), "mask_identical": bool(torch.equal(Ma, Mb)),
                         "seconds": round(time.time() - t0, 1)}
    if not (report["gallery"]["bank_identical"] and report["gallery"]["mask_identical"]):
        failures.append("fused gallery bank or mask differs")
    Xa, Ma, Xb, Mb = (t.to(dev) for t in (Xa, Ma, Xb, Mb))

    per_set, recall_src, recall_exp, ranked = {}, {}, {}, {}
    for qs in query_sets:
        packed = PackedQueries(*paths.packed_queries(args.split, qs))
        Q, Qm = packed.single_caption(slice(0, len(packed)))
        Q, Qm = torch.from_numpy(Q).to(dev), torch.from_numpy(Qm).to(dev)
        Sa = maxsim_scores(Q, Qm, Xa, Ma, query_chunk=args.query_chunk, gallery_chunk=args.gallery_chunk)
        Sb = maxsim_scores(Q, Qm, Xb, Mb, query_chunk=args.query_chunk, gallery_chunk=args.gallery_chunk)
        oa = torch.sort(Sa, dim=1, descending=True, stable=True).indices
        ob = torch.sort(Sb, dim=1, descending=True, stable=True).indices
        top10 = (oa[:, :10] == ob[:, :10]).all(dim=1)
        rel = relevance_indices(packed.ids, gallery)
        ra, _ = relevant_ranks(Sa, rel)
        rb, _ = relevant_ranks(Sb, rel)
        recall_src[qs], recall_exp[qs] = recall_at_k(ra, KS), recall_at_k(rb, KS)
        per_set[qs] = {"queries": len(packed), "scores_identical": bool(torch.equal(Sa, Sb)),
                       "max_abs_score_diff": float((Sa - Sb).abs().max()),
                       "full_rankings_identical": bool(torch.equal(oa, ob)),
                       "top10_identical_queries": int(top10.sum()), "relevant_ranks_identical": bool(torch.equal(
                           torch.as_tensor(ra), torch.as_tensor(rb)))}
        ranked[qs] = (packed.ids, oa[:, :10].cpu(), ob[:, :10].cpu())   # top-10 only: small host memory
        if not (per_set[qs]["scores_identical"] and per_set[qs]["full_rankings_identical"]):
            failures.append(f"{qs}: scores or rankings differ")
        print(f"[{qs}] n={len(packed)} scores identical={per_set[qs]['scores_identical']} "
              f"max|diff|={per_set[qs]['max_abs_score_diff']:.3g} rankings identical="
              f"{per_set[qs]['full_rankings_identical']}", flush=True)
    report["queries"] = {"sets": per_set, "total": sum(v["queries"] for v in per_set.values()),
                         "all_scores_identical": all(v["scores_identical"] for v in per_set.values()),
                         "all_rankings_identical": all(v["full_rankings_identical"] for v in per_set.values())}
    report["recall_recomputed"] = {
        "note": "consistency check of the two files; not a new reported result",
        "source_macro": macro_average(recall_src, KS), "export_macro": macro_average(recall_exp, KS),
        "identical": recall_src == recall_exp}
    if recall_src != recall_exp:
        failures.append("recomputed recall differs")

    if args.expected_rankings:
        data = json.loads(Path(args.expected_rankings).read_text(encoding="utf-8"))
        checks = []
        for ex in data["examples"]:
            ids, oa, ob = ranked.get(ex["query_set"]) or (None, None, None)
            if ids is None:
                checks.append({"image_id": ex["image_id"], "query_set": ex["query_set"], "checked": False})
                failures.append(f"{ex['query_set']} not evaluated for expected ranking {ex['image_id']}")
                continue
            row = ids.index(ex["image_id"])
            want = [t["image_id"] for t in ex["top5"]]
            got_a = [gallery[j] for j in oa[row, :5].tolist()]
            got_b = [gallery[j] for j in ob[row, :5].tolist()]
            ok = got_a == want == got_b
            checks.append({"image_id": ex["image_id"], "query_set": ex["query_set"], "recorded_top5": want,
                           "source_top5": got_a, "export_top5": got_b, "match": ok})
            if not ok:
                failures.append(f"recorded top-5 differs for {ex['image_id']} / {ex['query_set']}")
        report["expected_rankings"] = {"file_sha256": sha256_file(args.expected_rankings), "checks": checks}

    report["settings"] = {"precision": "float32", "tf32": {"cuda_matmul": torch.backends.cuda.matmul.allow_tf32,
                                                          "cudnn": torch.backends.cudnn.allow_tf32,
                                                          "float32_matmul_precision": torch.get_float32_matmul_precision()},
                          "tie_policy": TIE_POLICY, "bank_batch": args.bank_batch, "query_chunk": args.query_chunk,
                          "gallery_chunk": args.gallery_chunk, "query_sets": query_sets}
    report["environment"] = environment_info()
    report["failures"] = failures
    report["passed"] = not failures
    write_json_new(args.output, report)
    print("PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    print(f"Wrote {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
