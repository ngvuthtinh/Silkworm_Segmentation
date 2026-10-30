# 📝 multitask_vmunet_spatialemb — baseline + spatial embeddings (offset to the larva's centre)

## Goal
Same problem as `multitask_vmunet_disc` (L1: a larva hidden by a leaf into several pieces is **one id** in the labels but one region per connected piece for the watershed), tried with a different embedding family:
**spatial embeddings** (Neven et al., CVPR 2019, arXiv:1906.11109). Every pixel predicts an offset pointing to the centre of *its larva*; pixels of the same larva therefore land near one point **whether or not they touch**, and a learned per-larva margin (σ) decides which pixels belong to it. The **watershed path of the baseline is kept** and evaluated on the same predictions, so the comparison is within one model.

> **Rule for this experiment: one id per larva, every fragment shares that id** (one colour in the pictures). The loss uses the full multi-piece mask of each larva; clustering gives all pieces of a larva one id.

## Names (renamed on 2026-10-27 so they cannot be confused)
| Folder | Meaning | Run |
|---|---|---|
| `multitask_vmunet` | baseline B1: 3 channels, watershed | `2026-10-14_23-36-58` |
| `multitask_vmunet_disc` (was `_emb`) | **disc**riminative loss (De Brabandere 2017) + `merge_fragments`; classes `DiscConfig`, `DiscVMUNet` | `2026-10-27_02-26-27` (E1) |
| `multitask_vmunet_spatialemb` (this) | **spatial** embeddings (Neven 2019) + clustering; classes `SpatialEmbConfig`, `SpatialEmbVMUNet` | not trained yet |

## What is different from the baseline
| | Baseline | This experiment |
|---|---|---|
| Output channels | 3 (body, boundary, disease) | 3 + 2 offset + `n_sigma` σ + 1 seed = **8** (default `n_sigma = 2`) |
| Loss | `L_body + L_bnd + L_dis` | same + `λ_spatial · (w_inst·L_inst + w_var·L_var + w_seed·L_seed)` |
| Tách con | `split_instances` (watershed) | both on the same prediction: watershed (as baseline, `*_watershed` columns) **and** `cluster_spatial_embeddings` (main columns) |
| Everything else (data, lr, epochs, batch, seed, λ_body/bnd/dis, thresholds) | — | **identical**: `SpatialEmbConfig` inherits `MultiTaskConfig` |

> ⚠️ **One-change rule.** Like `_disc`, new output channels (architecture) and a new loss cannot be separated; record it as one experiment and say so in the `Đổi gì` cell of `Reports/experiment_results.xlsx`.

Still one forward pass, one summed loss, one optimiser step (multi-task, not multi-stage). The boundary channel is still trained so the watershed path stays comparable.

## Method (what was read and what was ported)
Read: the paper and `models/SpatialEmbeddings` (`my_loss.py`, `utils.Cluster`, `BranchedERFNet.init_output`, `train_config.py`). **Not run as-is** (PyTorch 1.1, ERFNet, Cityscapes; its `lovasz_hinge` even fails on current PyTorch because of a subtraction on a bool tensor). Only the loss and the clustering were ported:
- `e_i = tanh(offset_i) + x_i` (x = coordinates normalised to [0, 1]² for the square 256² input).
- `φ_k(e_i) = exp(−Σ (e_i − C_k)² · exp(10·σ̄_k))`; σ̄_k = mean σ over larva k; `n_sigma = 2` gives an elliptical margin (paper Table 2: better than circular; larvae are elongated).
- `L_inst`: Lovász-hinge (surrogate of 1 − IoU) of `φ_k` against the **whole** mask of larva k; `L_var`: σ constant inside a larva (weight 10); `L_seed`: seed regresses to `φ_k` inside larvae (weight 10) and to 0 on background; weights (1, 10, 1) as in the repo.
- Centre `C_k`: `to_center=False` (default) = mean of the larva's *embeddings* (learnable centre of attraction, paper Table 2: slightly better); `True` = mean of coordinates. **Why False**: a curved or leaf-split larva has its geometric centroid in the gap or outside the body, and background pixels *at* the centroid would look like the larva.
- Clustering (`src/metrics.py::cluster_spatial_embeddings`): foreground = body channel ≥ `body_thr`; repeatedly take the unassigned pixel with the highest seed (≥ `seed_thr`), use its embedding as centre and its σ as margin, and give every pixel with `φ > 0.5` one id. Differences from the repo: foreground from the body channel (not seed > 0.5), only unassigned pixels are taken, area threshold in 256² pixels (`cluster_min_area = 30`).
- Output layer init as the repo: offsets 0, raw σ = 1.
- `src/losses.py::SpatialEmbeddingLoss` is a **vectorised** rewrite over [image × larva × pixel] (the Python loop over larvae was 316 ms/step, now 37 ms). Checked in `test_spatialemb.py` against the repo's own loss (n_sigma 1/2, both centre modes, with disjoint fragments sharing an id) and Lovász against the repo's function: equal to numerical precision.

## ⚠️ License
The repo is **CC BY-NC 4.0** (research and personal use only, attribution required; commercial licence from the authors otherwise). The code here is a re-implementation from the paper and the repo's formulas, with attribution in `src/losses.py`, but **using it in a commercial production line needs the authors' permission**. Fine for the thesis; mention it if the project is ever deployed commercially. (`_disc` follows the 2017 paper; the Wizaron repo it was compared with is GPLv3 and is not used.)

## Run
```bash
python -m experiments.multitask_vmunet_spatialemb.test_spatialemb                 # unit checks, no GPU
python -m experiments.multitask_vmunet_spatialemb.train --smoke-test --gpu 1      # 5 train + 3 val batches
python -m experiments.multitask_vmunet_spatialemb.train --gpu 1 --out-dir Reports/Meeting_2026-10-30/checkpoints/multitask_vmunet_spatialemb
# choose seed_thr on valid ONLY, then test once:
python -m experiments.multitask_vmunet_spatialemb.evaluate --checkpoint <run>/checkpoints/best.pth --split valid --seed-thr 0.4
python -m experiments.multitask_vmunet_spatialemb.evaluate --checkpoint <run>/checkpoints/best.pth --split test
python -m experiments.multitask_vmunet_spatialemb.evaluate --checkpoint <run>/checkpoints/best.pth --split test_real
# pictures, one colour per larva (matched predictions use the colour of their ground-truth larva):
python -m experiments.multitask_vmunet_spatialemb.inference --checkpoint <run>/checkpoints/best.pth --split test --num 12 --output-dir <dir> --gpu 0
# all five ways side by side (B1 / disc / spatial, incl. leaf-split larvae by number of pieces):
PYTHONPATH=. python Reports/Meeting_2026-10-30/results/comparison/source/compare.py --spatial-ckpt <run>/checkpoints/best.pth --splits test test_real --gpu 1
```
Run training in `tmux`. Time: the `disc` run took ≈ 5.5 h for 40 epochs on the shared GPUs; measured per-step cost here is ≈ 25 % higher (loss 37 ms + larger backward) and validation adds a clustering step, so expect roughly **7–8 h** with the same sharing (not measured end to end).

## Reading `train_log.csv` (new columns)
| Column | Meaning |
|---|---|
| `loss_spatial`, `loss_sp_inst`, `loss_sp_var`, `loss_sp_seed` | the new loss and its three terms (`inst` = 1 − IoU surrogate: low means larva masks are recovered) |
| `inst_f1`, `e2e_f1` … | spatial-embedding path (**main score**, used to pick `best.pth`) |
| `*_watershed` (`inst_f1_watershed`, `e2e_f1_watershed`, `grasserie_recall_watershed` …) | the baseline watershed on the **same** model; their difference is the effect of the new separation step |
| `n_regions_spatial`, `n_regions_watershed` | total predicted regions over the validation set |

## Risks / things to watch
- **`seed_thr`** (0.5 default) and `n_sigma`/`to_center` are untuned guesses; `seed_thr` must be tuned on `valid`, never on `test`. If `e2e_f1` is far below the watershed column, look at the seed map panel in `inference.py` first (no seed above the threshold = no larva).
- **Generalisation (L2)**: the discriminative-loss paper warned about this family on varied scenes; check the two out-of-domain photos in `data/test` too.
- Like `_disc`, composed images use synthetic polygon leaves; `test_real` cannot show L1 (every connected blob is one larva there). See `Reports/Meeting_2026-10-30/results/README.md`.
- **Loss scale**: `loss_spatial` at initialisation is ≈ 2.5 against ≈ 1.5 for body/boundary; `_disc` showed that a large extra gradient can cost `grasserie_recall`. If it drops, try `--lambda-spatial 0.3`.
- Pretrained VMamba weights are not present (same as the baseline): training from scratch.
- Baseline B1 was trained on the 20,000-image version before the 11-image fix; this run uses `data/sam3_aug20k_v3` (19,989).

## Results table
Not trained yet. After training + evaluation: save scores as `eval_*.txt` under `results/<run>/eval_data_<x>/`, run `python utils/export_results_xlsx.py`, then fill the three manual columns (`Mã`, `Đổi gì`, `Ghi chú`) yourself.
