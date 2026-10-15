# 📝 multitask_vmunet — Per-larva segmentation + per-larva diagnosis

## Goal
Production line: for every photo, find each larva (even touching / overlapping ones) and say whether **that larva** is Healthy or Grasserie.

## Design
- **Model**: VM-UNet (VSSM) with 3 output channels — body, boundary, disease. One shared encoder–decoder = multi-task.
- **Loss**: `λ_body·(BCE+Dice) + λ_boundary·(BCE+Dice) + λ_disease·BCE(inside body)`.
- **Post-processing** (`src/metrics.split_instances`): seeds = body − boundary → watershed over body → one region per larva → mean disease prob → diagnosis.
- **Data**: `data/sam3_aug20k_v3` (built from `sam3_seg_v3`: scene-based split, SAM 3 masks fixed by a rescan, ignore regions) — train 20,000 (6,872 real + 13,128 composed) / valid 1,000 / test 1,000 composed + valid_real 505 / test_real 499 real photos.
- **Model selection**: best validation `e2e_f1` (separated with IoU ≥ 0.5 *and* correct diagnosis).

## Upper bound of the post-processing
Ground-truth maps fed as "perfect predictions" (min_area = 100 px at 256², ignore regions excluded), on `sam3_aug20k_v3`:

| Split | Larvae (one id each) | Separation F1, one id per larva | Separation F1, each visible piece = a larva |
|---|---|---|---|
| valid | 1,975 | 0.799 | – |
| test | 1,916 | 0.794 | 0.990 (2,770 pieces) |
| valid_real | 515 | 1.000 | – |
| test_real | 516 | 1.000 | – |

A larva hidden by a leaf into several pieces is **one id** in `masks_inst`, but `split_instances` returns **one region per connected piece**, so even perfect maps cannot reach 1.0 on composed images. (An older version of this table gave 0.97 because it counted every piece as a larva; `evaluate.py` does not.) A trained model cannot beat these numbers with the current post-processing. Counting pieces instead of larvae, the baseline reaches e2e_f1 0.886 on `test` (0.743 with one id per larva).

## Environment
Needs the `mamba_ssm` CUDA kernel (installed in `.venv`, torch 2.4.1 + cu124). Benchmark on A100 (shared), 256², AMP:

| Scan implementation | Batch | s/step | VRAM | 40 epochs (20k imgs) |
|---|---|---|---|---|
| PyTorch fallback | 2 | 18.0 | 8.7 GB | ~83 days |
| PyTorch fallback | 4 | 21.8 | 17.0 GB | ~50 days |
| CUDA kernel | 8 | 0.10 | 3.4 GB | ~2.8 h |
| CUDA kernel | 16 | 0.19 | 6.3 GB | ~2.6 h |

## Runs
Checkpoints: `Reports/Meeting_2026-10-16/checkpoints/multitask_vmunet/<run>/`; results: `Reports/Meeting_2026-10-16/results/<run>/` (see `results/README.md`). Scores are `e2e_f1` on the data named in the column; **only compare numbers measured on the same data**.

| Run | Trained on | Notes | valid | test | valid_real | test_real |
|---|---|---|---|---|---|---|
| 2026-10-03_23-04-46 | – | 1 epoch, pipeline check | – | – | – | – |
| 2026-10-03_23-12-56 | `sam3_aug20k_v1` (leaky) | 40 ep, best ep 31 | 0.872 | 0.847 | – | 0.899 |
| 2026-10-14_00-01-58 | `sam3_aug20k` (clean split, masks not fixed) | 40 ep, best ep 29; baseline B0 | 0.736 (own valid) | 0.709 (own test) | – | 0.970 (own test_real) |
| same run | scored on `sam3_aug20k_v3` | fair comparison | 0.742 | 0.729 | 0.986 | 0.984 |
| 2026-10-14_23-36-58 | `sam3_aug20k_v3` | 40 ep, best ep 34; current baseline B1 | 0.747 | 0.743 | 0.990 | 0.983 |

⚠️ The 2026-10-03 run used `sam3_aug20k_v1`: the Roboflow split leaked (~80% of valid/test images had a same-scene image in train), so its scores are **optimistic and not comparable** with the later runs. Fixing the masks (B0 → B1) changed the scores by 0.005–0.014 on composed images and by ≤ 0.004 on real photos (one seed each, so differences below ~0.01 are not conclusive).

## Limitations of the baseline (summary)
Details, evidence and caveats: `Reports/Meeting_2026-10-16/results/limitations_summary.md`.
- Larvae hidden into several pieces are split into several regions (structural, see the ceiling above).
- Photos unlike the training photos (small dense larvae on leaves) give almost no detection (probability of body ≤ 0.07), while SAM 3 segments them.
- On real single-larva photos all 13 errors out of 981 are Grasserie larvae (misdiagnosed, missed, or an extra region).
- Training and unseen scores are equal on composed images (0.737 vs 0.743): the model is not memorising; the limit is the method and the diversity of the 587 real scenes, not the number of images.

## Open questions
- Is `boundary_width = 1` enough to separate larvae lying side by side? Try 2 if merges dominate.
- Does the boundary channel need a higher weight (`lambda_boundary`)?
- Real multi-larva photos with per-larva disease labels are needed for a trustworthy test.
