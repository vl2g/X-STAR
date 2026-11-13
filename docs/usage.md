# Installation and usage

Run the commands below from the repository root. Inputs and model files must already be available; this guide does not provide downloads.

## Installation

Python 3.10 or newer. Inside a virtual environment:

```bash
pip install -e .                                        # numpy, torch, pyyaml
pip install -e ".[features]"                            # pillow, ftfy, regex (feature extraction)
pip install "git+https://github.com/openai/CLIP.git"    # original OpenAI CLIP (feature extraction, text queries)
pip install -e ".[test]"                                # pytest
```

Training, evaluation and retrieval with cached queries need only the base
dependencies. Feature extraction and free-text retrieval use OpenAI CLIP
ViT-B/32; its weights are read from `~/.cache/clip/ViT-B-32.pt` (or
`--clip-download-root`) and are downloaded by the `clip` package only when
`--allow-download` is given. Importing `xstar` or running any script with
`--help` loads no model.

Install a CUDA build of PyTorch yourself if you want GPU runs; every script
defaults to the CPU and uses a GPU only when given `--device cuda:N`.


## Workflow

```
  separately obtained source images (BSTD, RCTW-17)
                 │  prepare_dataset.py      fixed manifest records + dataset/rctw17_chinese_mapping.csv
                 ▼
  X-STAR image layout + manifest ── validate_dataset.py
                 │
                 ├─ extract_image_features.py   CLIP ViT-B/32 visual tokens [N, 50, 512] (+ global [N, 512])
                 ├─ extract_query_features.py   per-token queries [N, 1, 77, 512] + masks + IDs,
                 │                              global queries [N, 512] for the CLIP baseline
                 └─ extract_ocr_features.py     one CLIP embedding per supplied English OCR line
                 ▼
  train.py      TextCaps pre-training → X-STAR fine-tuning (from frozen features)
  evaluate.py   full-gallery Recall@1/5/10 for all 13 query sets, CoLITE-CLIP and CLIP baseline
  retrieve.py   ranked images for one English query from a checkpoint and a fused gallery
```

OCR recognition (IndicPhotoOCR) and translation are **upstream preprocessing**.
The pipeline starts from the supplied English OCR lines and the supplied
back-translated English queries; it calls no OCR engine and no translation
service. Recognised OCR and ground-truth OCR are separate variants
(`--ocr-field`, `features.ocr.variant`) and are never mixed.


## Usage

Copy [`configs/paths.example.yaml`](../configs/paths.example.yaml) to
`configs/paths.yaml` (ignored by git) and fill in the locations of the
manifest, split lists, feature cache and checkpoints. Every script has
`--help`; outputs go to new directories and are refused if they overlap an
input.

```bash
# Check the manifest, splits, images and cached features.
python scripts/validate_dataset.py --paths configs/paths.yaml --check-features

# Evaluate a checkpoint (a file, or a key under 'checkpoints:') on all 13 query sets.
python scripts/evaluate.py --paths configs/paths.yaml --checkpoint xstar_main \
    --output-dir runs/eval_main --experiment-name main_checkpoint

# Rank test images for an English query, or for a supplied X-STAR query.
python scripts/retrieve.py --paths configs/paths.yaml --checkpoint xstar_main \
    --query "Lokpriya Gopinath Bordoloi airport exterior entrance" --top-k 5
python scripts/retrieve.py --paths configs/paths.yaml --checkpoint xstar_main \
    --query-image-id A_image_103 --query-set hindi_to_english --top-k 5

# Fine-tune on X-STAR from the TextCaps checkpoint (stage settings in the config).
python scripts/train.py --config configs/xstar_finetune.yaml --paths configs/paths.yaml \
    --output-dir runs/xstar_finetune --device cuda:0

# Extract features for one split from the manifest and images.
python scripts/extract_image_features.py --manifest /path/to/XSTAR_final_manifest2.json \
    --image-root /path/to/XSTAR_dataset --split test --global --output-dir features/images_test
python scripts/extract_query_features.py --manifest /path/to/XSTAR_final_manifest2.json \
    --split test --output-root features/queries_test
python scripts/extract_ocr_features.py --manifest /path/to/XSTAR_final_manifest2.json \
    --split test --output-dir features/ocr_test_vitb32_lines
```

`evaluate.py` writes `<experiment>__ckpt-<sha12>__<split>__complete__eval.json`
(or `__partial__` when `--query-sets` or `--max-queries` restrict the run) with
per-set and macro-averaged recalls, query counts, gallery size, checkpoint
SHA-256, manifest identity, feature locations, settings and package versions.
`train.py --resume` continues a run from one of its own epoch checkpoints;
`--init-checkpoint` only loads weights. Dataset preparation is described in
[`dataset/README.md`](../dataset/README.md#preparation).

Inference-only checkpoints: `scripts/export_inference_checkpoint.py` writes a
compact copy of a training checkpoint (about a third of the size) with
bit-identical model weights and no optimiser or resume state. Evaluate and
retrieve accept it like any checkpoint; training cannot resume from it.
Commands and verification: [`inference_checkpoints.md`](inference_checkpoints.md).

```bash
python scripts/export_inference_checkpoint.py --input /path/to/best.pt \
    --experiment my_experiment --split original --tau-mode stored_parameter \
    --output runs/inference_checkpoints/my_model__inference.pt
```

See [dataset integrity](dataset_integrity.md) for original-split overlap and tie handling.

Tests: `pytest`. Setting `XSTAR_RESEARCH_ROOT` to a local copy of the research
storage also runs the checks against the real checkpoint and features. Tests
that use DataLoader worker processes report `UNVERIFIED` where the environment
blocks them (`python tools/check_dataloader_workers.py` checks this).


## Auditing a local copy of the research artifacts

[`tools/audit_release_inputs.py`](../tools/audit_release_inputs.py) reads an
existing X-STAR layout, never modifies it, and writes a JSON report inside this
repository:

```bash
python tools/audit_release_inputs.py \
    --dataset-root /path/to/XSTAR_dataset \
    --textcaps-checkpoint /path/to/textcaps_checkpoint/best.pt
```
