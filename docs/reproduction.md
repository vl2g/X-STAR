# Results and reproduction status

The main-checkpoint evaluation verifies retrieval from cached features. It does not establish a complete rerun from raw source data through TextCaps pre-training.

## Results (full reproduction pending)

The values below are rounded to one decimal place. Both rows come from one
complete run of the released evaluator (`scripts/evaluate.py`) with the
existing main checkpoint and the existing cached features: all 13 query sets,
16,757 queries, each scored against the full 1,289-image test gallery. They
equal the saved main-notebook output (CoLITE-CLIP) and the pre-release CPU
audit (CLIP baseline). Features were not regenerated and the model was not
retrained, so this verifies checkpoint evaluation, not a reproduction from raw
images and training. The paper draft will be updated later.

| Setting | R@1 | R@5 | R@10 | Source |
|---|---:|---:|---:|---|
| CoLITE-CLIP, X-STAR test, mean of 13 query sets | 46.5 | 65.8 | 73.0 | Released evaluator, main checkpoint, cached features |
| CLIP ViT-B/32 global baseline, X-STAR test, mean of 13 query sets | 24.6 | 42.8 | 52.0 | Released evaluator, cached global CLIP features |

Unrounded: 46.4642 / 65.8113 / 73.0083 and 24.6404 / 42.7941 / 51.9544. Per
language results, provenance and checksums are in
[`docs/evaluation/main_checkpoint_cached_features.md`](evaluation/main_checkpoint_cached_features.md).
Two pairs of identical test images produce tied scores; ties are ordered by the
fixed gallery order. Under other tie handling the rounded baseline R@1 or R@10
could read 24.7 or 51.9 (CoLITE-CLIP is unaffected after rounding); see the
evaluation report. The original multi-language evaluator contained a
path error that silently replaced the CLIP baseline's global query embeddings
with averaged token embeddings; `scripts/evaluate.py` refuses to run without the
intended global features (the main-model evaluator requires the intended global embeddings).

**Dataset integrity.** An audit of image bytes found 61 pairs of identical
image files in the original split: 15 test and 8 validation images are
identical to a training image, 2 validation/test pairs share an image, and 2
test images each have an identical copy in the test gallery. The results above
are for the original split as published; whether the overlap affects them has
not been measured. A perceptual-hash screen additionally flags suspected
near-duplicates (same scene photographed moments apart) across splits; these
are distinct photographs and remain separate records. See
[`docs/dataset_integrity.md`](dataset_integrity.md).

Artifact availability: [resources](resources.md).
