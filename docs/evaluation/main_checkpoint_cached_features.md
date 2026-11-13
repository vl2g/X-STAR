# Main checkpoint on cached features: complete X-STAR test evaluation

**Scope.** The existing main CoLITE-CLIP checkpoint was evaluated with the
released evaluator (`scripts/evaluate.py`) on the existing cached CLIP
features. This confirms the checkpoint-evaluation stage of the pipeline. It is
**not** a reproduction from raw images: no features were regenerated and no
model was trained.

Machine-readable summary: [`main_checkpoint_cached_features.json`](main_checkpoint_cached_features.json)
(the raw result file with machine-specific path prefixes replaced; all
numbers identical, see its `sanitization` block).

## Run

| | |
|---|---|
| Date (UTC) | 2026-10-22 |
| Code | historical research revision `81b6d5a00b544d4cb23e59f2e409ee1af5857342`, clean working tree (before this public snapshot) |
| Command | `CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --paths configs/paths.yaml --checkpoint xstar_main --output-dir runs/main_checkpoint_verified --experiment-name main_checkpoint_cached_features --device cuda:0` |
| Device | one NVIDIA RTX A6000 (idle before the run); 45 s wall time |
| Precision | float32; TF32 disabled (`cuda.matmul.allow_tf32 = False`, `cudnn.allow_tf32 = False`, matmul precision `highest`) |
| Chunking | fusion batch 256 images; MaxSim 32 queries × 128 images |
| Ties | `stable_gallery_order`: equal scores keep the order of the ordered test image list |
| Software | Python 3.10.19, PyTorch 2.4.1+cu121, NumPy 2.2.6, xstar 0.1.0.dev0 |

## Inputs

| Artifact | Identity |
|---|---|
| Checkpoint `checkpoints_colbert_XSTAR_5/best.pt` | SHA-256 `b702d7eb1017771ed3aa67120cbed39d5900fa6192da3244b469ea57680977b8`, epoch 9, K = 40, `stored_parameter` τ 33.3333 |
| Manifest `XSTAR_final_manifest2.json` | SHA-256 `f6e594ac5d0eb470c5ff32bbc4637c5831c1cf2727e9227e2f023cde457d26a2` |
| Test image list `image_filenames_test.txt` | SHA-256 `125cf37abc8897db8e4da030ea18c7a5dd1225e88556ca84c318bfca7c503c36`; ordered gallery IDs SHA-256 `65ae7c10e01336f85ae6e3a87f5a1c5436c95a5ea980d04d1ed21522bb29d283` |
| Visual tokens / global image features | `test_image_tokens_postxfmr_vitb32.npy`, `test_image_global_embeddings.npy` |
| Query features | `colbert_packed2/` (CoLITE-CLIP), `english_query_vitb32_2/` and `to_english_vitb32_2/` (CLIP baseline); every set's packed and global ID files have SHA-256 `3862fc73738700fb…` |
| OCR | recognized OCR, `ocr_test_vitb32_lines/` (not ground-truth or corrected OCR) |

## Coverage

Derived from the manifest before the model was loaded: 1,289 test images, and
in each of the 13 query sets exactly one query per test image (no missing,
extra or duplicated IDs; packed and global IDs aligned). 16,757 queries in
total, every set scored against the full 1,289-image gallery. The result is
marked `complete_benchmark: true`.

## Results (Recall@K, %)

| Query set | Queries | CoLITE-CLIP R@1 | R@5 | R@10 | CLIP R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| english_query | 1,289 | 52.21 | 69.28 | 75.72 | 27.08 | 45.54 | 55.55 |
| assamese_to_english | 1,289 | 44.22 | 62.06 | 70.13 | 23.58 | 41.82 | 51.82 |
| bengali_to_english | 1,289 | 45.07 | 65.01 | 72.92 | 24.20 | 42.44 | 51.05 |
| gujarati_to_english | 1,289 | 47.01 | 67.96 | 74.55 | 24.52 | 43.06 | 52.75 |
| hindi_to_english | 1,289 | 46.78 | 66.95 | 72.92 | 24.90 | 43.21 | 52.29 |
| kannada_to_english | 1,289 | 45.85 | 64.70 | 72.30 | 24.28 | 42.98 | 52.21 |
| malayalam_to_english | 1,289 | 48.95 | 67.65 | 74.94 | 25.37 | 43.99 | 51.98 |
| marathi_to_english | 1,289 | 42.98 | 63.54 | 70.44 | 23.58 | 41.19 | 49.57 |
| odia_to_english | 1,289 | 46.86 | 65.71 | 73.62 | 24.98 | 41.89 | 51.59 |
| punjabi_to_english | 1,289 | 46.39 | 66.18 | 73.08 | 24.44 | 42.20 | 52.06 |
| tamil_to_english | 1,289 | 43.83 | 65.09 | 73.16 | 23.89 | 41.82 | 51.05 |
| telugu_to_english | 1,289 | 46.39 | 65.71 | 73.16 | 24.52 | 43.21 | 51.28 |
| chinese_to_english | 1,289 | 47.48 | 65.71 | 72.15 | 24.98 | 42.98 | 52.21 |
| **Macro average** | 16,757 | **46.4642** | **65.8113** | **73.0083** | **24.6404** | **42.7941** | **51.9544** |

The CLIP baseline uses global CLIP query embeddings against global CLIP image
embeddings; no token or patch averages are involved.

## Comparison with the references

| | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| CoLITE-CLIP, this run | 46.4642 | 65.8113 | 73.0083 |
| CoLITE-CLIP, saved main-notebook output | 46.4642 | 65.8113 | 73.0083 |
| CLIP global baseline, this run | 24.6404 | 42.7941 | 51.9544 |
| CLIP global baseline, pre-release CPU audit | 24.6404 | 42.7941 | 51.9544 |

Per query set and metric, every CoLITE-CLIP recall equals the saved notebook
value (largest difference 3.4 × 10⁻⁶ percentage points, the float32
representation of the saved fractions: the same number of hits in all 39
cells), and every CLIP recall equals the audit value exactly. The saved
notebook's CLIP column is not a reference: it used the token-average fallback
for the 12 back-translated sets.

## Ties and duplicate images

Every query set has exactly 4 queries (for both models) whose relevant image
ties with another gallery image. They come from two pairs of byte-identical
test images stored under different source folders: `A_image_3441` /
`B_image_50` and `B_image_287` / `D_image_1404` (same image file, OCR file and
features). Under the documented tie policy the earlier image in the list wins.

Bounds over tie handling (CPU recomputation; the gallery-order values equal
this run):

| | Policy | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| CoLITE-CLIP | all ties won | 46.5477 | 65.8173 | 73.0083 |
| | gallery order (reported) | 46.4642 | 65.8113 | 73.0083 |
| | all ties lost | 46.4522 | 65.8113 | 73.0083 |
| CLIP | all ties won | 24.7121 | 42.8000 | 51.9544 |
| | gallery order (reported) | 24.6404 | 42.7941 | 51.9544 |
| | all ties lost | 24.6404 | 42.7583 | 51.9484 |

The rounded CoLITE-CLIP values are the same under every policy; the rounded
CLIP R@1 (24.6 vs 24.7) and R@10 (52.0 vs 51.9) depend on how these ties are
broken.

A wider check of the cached visual tokens found 61 pairs of byte-identical
images in the dataset (122 images, mostly folders A/B): 2 pairs inside the test
split, 34 inside train, and 25 across splits; 15 test images are identical to a
training image and 2 to a validation image. Each pair has identical recognized
OCR lines; their English queries are identical in 4 of the 61 pairs and
different (generated per record) in the others. The complete byte-level audit in [`../dataset_integrity.md`](../dataset_integrity.md)
confirms these counts and documents the tie policy. No records were changed, and the effect on the results was not measured.

## Portable annotation download

The release removes server paths without changing record order, queries, OCR text, image IDs or split assignments. Its file hash differs from the historical evaluation input. Both identities and exact changes are recorded in [annotation_release.json](../../dataset/annotation_release.json). The checkpoint bytes and three ordered split lists are unchanged.
