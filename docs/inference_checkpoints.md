# Inference and training checkpoints

The main X-STAR model has two checkpoint forms:

| File | Size | Purpose |
|---|---:|---|
| Full training checkpoint | about 556 MB | Retains model weights and optimizer state for compatible training continuation |
| Compact inference checkpoint | about 185 MB | Evaluation, retrieval and initialization of a new training run |

The compact file is `xstar_main_original__inference__src-b702d7eb1017.pt`.
SHA-256: `e4e8a9a5ebd6cfc9acbacd553ce0fd5ffe791a8bff33c5839227b83728d56862`.
The source checkpoint SHA-256 is `b702d7eb1017` (prefix; full identity is in the evaluation report).

## Verified equivalence

All 354 model tensors are identical to the full checkpoint. On all 16,757 queries against the 1,289-image gallery, scores and rankings match exactly (maximum score difference 0). Recall@1/5/10 is 46.4642 / 65.8113 / 73.0083. All five saved qualitative rankings match. Checks used float32 on GPU 0 with TF32 disabled and fixed gallery-order ties.

The export records code revision `10d0f9b` with a dirty flag from concurrent README/license edits; those edits did not change exporter or model code. Raw export and equivalence reports remain with the checkpoint files in ignored storage.

## Continuing training

Keep the full training checkpoints on server storage. Compact exports omit optimizer state and resume metadata, so `train.py --resume` rejects them. `--init-checkpoint` loads their weights into a new run with a fresh optimizer and schedule.

Full checkpoints written by this package's trainer support `--resume` when the original run directory, inputs and settings match. They restore optimizer, epoch, global step, best validation value and random state. Use the latest completed epoch checkpoint to continue the interrupted run; `best.pt` may belong to an earlier epoch.

The original main checkpoint was produced by the legacy notebook. Its optimizer state can support continuation with compatible notebook training code, but the package's strict `--resume` requires its own run signature and metadata. Do not claim exact continuation from the legacy checkpoint through the new trainer.

## Downloads and use

Public weight downloads and artifact-specific terms are forthcoming; see [resources](resources.md#checkpoints). No checkpoint binary is included in Git. Local compact exports can be evaluated using:

```bash
python scripts/evaluate.py --paths configs/paths.yaml \
    --checkpoint /path/to/xstar_main_original__inference__src-b702d7eb1017.pt \
    --output-dir runs/eval_main_inference --experiment-name main_inference --device cuda:0
```

See [usage](usage.md) and [evaluation provenance](evaluation/main_checkpoint_cached_features.md).
