# 📝 multitask_vmunet — Per-larva segmentation + per-larva diagnosis

## Goal
Production line: for every photo, find each larva (even touching / overlapping ones) and say whether **that larva** is Healthy or Grasserie.

## Design
- **Model**: VM-UNet (VSSM) with 3 output channels — body, boundary, disease. One shared encoder–decoder = multi-task.
- **Loss**: `λ_body·(BCE+Dice) + λ_boundary·(BCE+Dice) + λ_disease·BCE(inside body)`.
- **Post-processing** (`src/metrics.split_instances`): seeds = body − boundary → watershed over body → one region per larva → mean disease prob → diagnosis.
- **Data**: `data/sam3_aug20k` (train 20k / valid 1k / test 1k synthetic + test_real 497).
- **Model selection**: best validation `e2e_f1` (separated with IoU ≥ 0.5 *and* correct diagnosis).

## Upper bound of the post-processing
Ground-truth maps fed as "perfect predictions" (min_area = 100 px at 256²):

| Split | Larvae | Separation F1 |
|---|---|---|
| valid | 720 | 0.972 |
| test | 764 | 0.938 |
| test_real | 336 | 0.882 (broken SAM 3 masks) |

A trained model cannot beat these numbers with the current post-processing.

## Environment
Needs the `mamba_ssm` CUDA kernel (installed in `.venv`, torch 2.4.1 + cu124). Benchmark on A100 (shared), 256², AMP:

| Scan implementation | Batch | s/step | VRAM | 40 epochs (20k imgs) |
|---|---|---|---|---|
| PyTorch fallback | 2 | 18.0 | 8.7 GB | ~83 days |
| PyTorch fallback | 4 | 21.8 | 17.0 GB | ~50 days |
| CUDA kernel | 8 | 0.10 | 3.4 GB | ~2.8 h |
| CUDA kernel | 16 | 0.19 | 6.3 GB | ~2.6 h |

## Runs
| Date | Run | Notes | valid e2e_f1 | test | test_real |
|---|---|---|---|---|---|
| 2026-10-03 | smoke_test | pipeline check only (5 batches) | – | – | – |

## Open questions
- Is `boundary_width = 1` enough to separate larvae lying side by side? Try 2 if merges dominate.
- Does the boundary channel need a higher weight (`lambda_boundary`)?
- Real multi-larva photos with per-larva disease labels are needed for a trustworthy test.
