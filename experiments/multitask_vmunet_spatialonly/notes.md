# 📝 multitask_vmunet_spatialonly — spatial embeddings with no boundary channel and no watershed

## Goal
Answer one question: **is the watershed path (and the boundary channel that only feeds it) still needed once larvae are separated by spatial embeddings?**
On the same trained `multitask_vmunet_spatialemb` model the spatial path beats the watershed path by +0.17 `e2e_f1` on composed images (test 0.907 vs 0.739) and is equal on real photos (0.986 vs 0.979). That says the *separation step* is not needed; it does not say whether *training* the boundary channel helps the other channels (shared encoder). This run removes it and keeps everything else.

## What is different from `multitask_vmunet_spatialemb`
| | spatialemb (E2) | **spatialonly (E3)** |
|---|---|---|
| Output channels | 3 + 2 + 2 + 1 = **8** (body, boundary, disease, offset×2, σ×2, seed) | 2 + 2 + 2 + 1 = **7** (body, disease, offset×2, σ×2, seed) |
| Loss | `L_body + L_bnd + L_dis + λ_sp·L_spatial` | `L_body + L_dis + λ_sp·L_spatial` (`BodyDiseaseLoss`; λ_bnd = 0) |
| Instance separation | spatial clustering (main) + watershed (`*_watershed`, comparison only) | spatial clustering only |
| Everything else (data, lr, epochs, batch, seed, λ_body, λ_dis, λ_sp, σ, `seed_thr`, thresholds, backbone) | — | **identical** (`SpatialOnlyConfig` inherits `SpatialEmbConfig`) |

> ⚠️ **One-change rule.** Removing the boundary channel changes the architecture (one output channel) *and* the loss (one term) at the same time; they cannot be separated. Record it as one experiment and say so in the `Đổi gì` cell of `Reports/experiment_results.xlsx`. The weights of the remaining terms are not changed, so the loss ratio shifts only because the boundary term is gone (see below).

Still one forward pass, one summed loss, one optimiser step (multi-task, not multi-stage). The dataset still builds a boundary map; it is ignored (cheap).

## Loss ratio (what the baseline weights mean in practice)
Weights are all 1 (`λ_body = λ_bnd = λ_dis = λ_spatial = 1`; inside the spatial term `w_inst = 1, w_var = 10, w_seed = 1`, seed foreground weight 10). The *share* of each term in the total is set by the size of the raw losses, from `train_log.csv` of run `2026-10-27_12-43-28` (SpatialEmb):

| Term | Epoch 1 (mean) | share | Epoch 40 (mean) | share |
|---|---|---|---|---|
| body (BCE+Dice) | 0.509 | 17 % | 0.038 | 7 % |
| **boundary (BCE+Dice)** | 1.023 | 34 % | **0.339** | **59 %** |
| disease (BCE inside body) | 0.366 | 12 % | 0.014 | 2 % |
| spatial (`inst` + 10·`var` + `seed`) | 1.106 | 37 % | 0.183 | 32 % |
| total | 3.004 | | 0.574 | |

The boundary term is the **largest** at the end of training (the thin 1-px contour is hard to fit) although the spatial path never uses it. In this run it disappears, so the spatial term becomes ≈ 78 % of what is left at epoch 40 (0.183 of ≈ 0.235) and the optimiser's gradient is no longer dominated by the contour. If the spatial path turns out better, that is one possible reason; if body/disease get worse, the contour was helping the encoder. (Not measured: this is the experiment that measures it.)

## Architecture (current SpatialEmb, and this variant)
```
RGB image 256×256
      │
VSSM encoder (VMamba, 4 stages, depths [2,2,2,2])            ← 27.4 M parameters, shared by every output
      │   skip connections
VSSM decoder (depths [2,2,2,1])
      │
final 1×1 conv  →  [B, C, 256, 256]   C = 8 (spatialemb)  /  7 (spatialonly)

 spatialemb :  ch0 body | ch1 boundary | ch2 disease | ch3-4 offset (x,y) | ch5-6 σ (x,y) | ch7 seed
 spatialonly:  ch0 body |               ch1 disease | ch2-3 offset (x,y) | ch4-5 σ (x,y) | ch6 seed
```
- **body**: is this pixel part of a larva? → also the region where clustering is allowed.
- **disease**: Grasserie probability per pixel; averaged over each larva's region → Healthy / Grasserie for that larva.
- **offset**: embedding of a pixel `e = tanh(offset) + (x, y)`; pixels of one larva are trained to land near one point, **touching or not** (that is why two leaf-split pieces of one larva get one id).
- **σ**: width of the larva's Gaussian margin (2 values = elliptical, fits elongated larvae). **seed**: how close the pixel is to its larva's centre.
- Last layer initialised as in the original repo: offset 0, raw σ 1 (so the model starts as "every pixel points at itself, margin e^-10σ").

Inference (`src/metrics.py::cluster_spatial_embeddings`): foreground = body ≥ 0.5 → take the unassigned pixel with the highest seed (≥ `seed_thr`) → its embedding is the centre, its σ the margin → every free pixel with φ > 0.5 gets that id → repeat. `seed_thr = 0.9` was chosen on `valid` for E2; the same value is the starting point here, but **re-tune it on `valid` for this run** before running `test`.

## Mixed-data runs (2026-10-28)
`--train-mix v3:1 dense_v2:8` (also on `multitask_vmunet_spatialemb`): `MixedScaleCropDataset` mixes `sam3_aug20k_v3` with `dense_seg_v2` (repeated 8×, ≈ 18 % of the samples, disease = unknown → no disease loss). Dense photos are rescaled to 0.5–1.0× native and a 256² window is cut; v3 is read exactly as before (checked bit-identical), so validation on `v3:valid` stays comparable with E2/E3. Run names: `..._mix`. Whole-image inference at a larger size: `evaluate.py` with a larger `--img-size` is still to be added/tested.

## Run
```bash
python -m experiments.multitask_vmunet_spatialonly.test_spatialonly                  # unit checks (needs a GPU for the model part)
python -m experiments.multitask_vmunet_spatialonly.train --smoke-test --gpu 1        # 5 train + 3 val batches (passed on 2026-10-28)
python -m experiments.multitask_vmunet_spatialonly.train --gpu 1 --out-dir Reports/Meeting_2026-10-30/checkpoints/multitask_vmunet_spatialonly   # in tmux
# choose seed_thr on valid ONLY, then test once:
python -m experiments.multitask_vmunet_spatialonly.evaluate --checkpoint <run>/checkpoints/best.pth --split valid --seed-thr 0.9
python -m experiments.multitask_vmunet_spatialonly.evaluate --checkpoint <run>/checkpoints/best.pth --split test
python -m experiments.multitask_vmunet_spatialonly.evaluate --checkpoint <run>/checkpoints/best.pth --split test_real
```
Time: the extra loss is the same as in `spatialemb` and one output channel less, so expect ≈ 5.5–6 h for 40 epochs on the shared GPUs (not measured).

## Reading `train_log.csv`
Same as `spatialemb` minus `loss_boundary` and the `*_watershed` columns; `inst_f1`, `e2e_f1`, … are the spatial path and `e2e_f1` picks `best.pth`.

## What to compare afterwards
- E2 vs E3 on the **same data and metrics** (`e2e_f1`, `grasserie_recall` on `test` and `test_real`): if E3 ≥ E2, the boundary channel and watershed can be dropped for good (simpler and slightly lighter model). If E3 < E2, the contour task was helping the shared encoder.
- Run the two photos in `data/test` too (CLAUDE.md §3.3): the question there is whether the body channel sees the larvae at all.

## Risks
- The boundary term currently dominates the total loss; removing it changes the effective learning rate of the other terms, so this is not a pure "ablation of one head". Compare with care.
- Single seed per model (as for E1, E2): differences of ≈ 0.01 are within noise.
- Same license caveat as `spatialemb` (CC BY-NC 4.0 reference code; research use).
- Pretrained VMamba weights are absent (training from scratch, as the baseline).

## Results (run `2026-10-28_01-21-50`, data v3, `seed_thr = 0.9` chosen on `valid`, `test` run once)
`e2e_f1`, spatial path of each model (E2 also has its watershed path):

| Split | E2 watershed | E2 spatial | **E3 spatial only** |
|---|---|---|---|
| valid | 0.740 | 0.928 | 0.929 |
| test | 0.739 | 0.907 | 0.901 |
| valid_real | 0.986 | 0.984 | 0.987 |
| test_real | 0.979 | 0.986 | 0.983 |

Reading: removing the boundary channel and the watershed costs nothing measurable (differences ≤ 0.006, inside the noise of one seed per model), and the spatial path is far better than the watershed on composed images (+0.16–0.19). Scores recorded in `Reports/Meeting_2026-10-30/results/spatialonly/scores/`, row added to `Reports/experiment_results.xlsx` (manual columns still to be filled by the user).

Dense photos in `data/test` (CLAUDE.md §3.3): E3 finds 0 larvae on both (body max prob 0.99 on `image copy.png`, 0.25 on `image.png`); no improvement over E2 (2 and 0).
