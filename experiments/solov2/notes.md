# 📝 solov2 — SOLOv2 with three backbones: original (ResNet-50), Mamba (VSSM encoder), VM-UNet

## Goal
Try a **different instance-segmentation family** next to the VM-UNet + embedding line (E1–E3), as decided on 17/10 ("stay open, try other models"). SOLOv2 (Wang et al., NeurIPS 2020, "Dynamic and Fast Instance Segmentation") predicts **one mask per larva directly from its location** (grid cell → dynamic kernel → mask), so a larva split into pieces by a leaf can still be one mask (L1), and every mask has its own **class** (Healthy / Grasserie) — per-larva diagnosis is built in. There is also a silkworm paper on it: *Enhanced SOLOv2 … Densely Overlapping Silkworms* (Sensors 2025, ref [23] of the scoping review).

Code: AdelaiDet (`models/AdelaiDet`, detectron2). **Nothing in `models/` is modified**; everything is wrapped here.

## The three variants (only the backbone changes)
| `--variant` | Backbone + neck | Params (total / backbone) | Pretrained |
|---|---|---|---|
| `r50` (original) | ResNet-50 + FPN (`build_resnet_fpn_backbone`) | 46.2 M / 26.8 M | ImageNet (MSRA R-50); `--no-imagenet` for from scratch |
| `mamba` (way A) | **VSSM encoder of VM-UNet** (Visual Mamba, 4 stages, strides 4/8/16/32, channels 96/192/384/768 = same slots as ResNet res2–res5) + FPN | 35.9 M / 16.5 M | none (no VMamba weights in the repo) |
| `vmunet` (way B) | **whole VM-UNet** (encoder + decoder with skips); the 4 decoder stages are used as p5..p2 (LayerNorm + 1×1 + 3×3 conv to 256 ch), p6 = max-pool(p5) | 49.5 M / 30.1 M | none |

VSSM depths are the ones of the VM-UNet baseline (`[2,2,2,2]` encoder, `[2,2,2,1]` decoder, drop-path 0.2). The SOLOv2 heads are unchanged (instance branch with 5 levels p2–p6, mask branch on p2–p5, matrix NMS).

```
image (short side 512)
  └─ backbone ─┬─ r50:    ResNet-50 res2..res5 ──► FPN ──────────────────► p2..p6
               ├─ mamba:  VSSM encoder res2..res5 (+LayerNorm) ──► FPN ──► p2..p6
               └─ vmunet: VSSM encoder ─► VSSM decoder (+skips) ─► LN+conv ► p2..p5, p6 = maxpool(p5)
  p2..p6 ─► instance branch: per grid cell → class score (Grasserie/Healthy) + dynamic kernel
  p2..p5 ─► mask branch: one mask feature map (1/4 size)
  mask of a cell = kernel ⊛ mask feature  → matrix NMS → one mask + class per larva
```

## Changes needed for our data (`model.py::SilkSOLOv2`, only label building + classification loss)
- **Unknown disease** (class 3 in `masks_cls`, e.g. all of `data/dense_seg_v2`): `category_id = -1`. The mask is learned as usual; in the classification branch the cell is taught "this is a larva of *some* class": focal loss on `p_any = 1 − Π_c(1 − σ(x_c))`, no class is forced. Without this, dense photos could not be used at all (SOLOv2 requires a class per instance).
- **Ignore regions** (`masks_ignore`): `category_id = -2`; no mask target, their grid cells are excluded from the classification loss (not taught as background).
- Upstream bug avoided: images with no larva crashed `get_ground_truth_single`.
- Checked in `test_solov2.py`: ignore cells get no gradient, unknown cells are pushed toward "larva", otherwise identical to the original focal loss.

## Training data of the first runs (2026-10-28, `utils/launch_mix_runs.sh`)
`--train-sets v3:train dense_v2:train*8` (default): `sam3_aug20k_v3` (labelled, Healthy / Grasserie) + `dense_seg_v2` repeated 8× (≈ 18 % of the samples; many larvae per photo, **no disease label** → unknown class, mask learned, no classification loss). Training: random short edge 400–880 (≈ 0.6–1.4× native), random **512² window cut at native scale**, flips; validation / test / inference run on the **whole image** (short edge 640, max 1280), no tiling. Model selection (`best.pth`) on `v3:valid`; report `dense_v2:valid/test` separately. 40,000 it, lr steps at 2/3 and 7/8.

## Training recipe (same for the three variants)
AdamW lr 1e-4, wd 0.05, warmup 1000 it (from 1 %), steps at 40k / 52k, 60k iterations, batch 8, short side 512, AMP, **full-model gradient clipping 1.0**.
> ⚠️ Without the full-model clipping (detectron2 only clips each parameter separately) and with a short warmup, AdamW made SOLOv2 blow up at step 1 (`loss_cate` 0.9 → 152, masks frozen). Found and fixed on 2026-10-28 with a one-batch test.

Overfit check (2026-10-28, 400 it on 64 train images, scored on 8 of them): all three variants reach `inst_f1 = e2e_f1 = 1.0` → labels, loss, inference and scoring are wired correctly. (The `vmunet` variant only learned masks after adding LayerNorm to the decoder outputs.)

## Scoring
`evaluator.py` scores exactly like the VM-UNet experiments (masks resized to 256², `min_area` 100, IoU ≥ 0.5, ignore regions not counted, unknown-disease larvae never count as a wrong diagnosis), so numbers are comparable with B1 / E1–E3. SOLOv2 masks with score ≥ `eval_score_thr` (0.3, **tune on valid**) are painted in increasing score order. `best.pth` = best `e2e_f1` on `v3:valid`.

## Run
```bash
python -m experiments.solov2.test_solov2                                     # unit checks (GPU)
python -m experiments.solov2.train --variant mamba --smoke-test --gpu 1     # 20 it, 2 img/batch, 8 valid images
# full runs (tmux), one per variant:
python -m experiments.solov2.train --variant r50    --gpu 1 --out-dir Reports/Meeting_<date>/checkpoints/solov2_r50
python -m experiments.solov2.train --variant mamba  --gpu 1 --out-dir Reports/Meeting_<date>/checkpoints/solov2_mamba
python -m experiments.solov2.train --variant vmunet --gpu 1 --out-dir Reports/Meeting_<date>/checkpoints/solov2_vmunet
# resume: ... --resume <run dir>      shorter run: --max-iter 40000      add the dense photos: --train-sets v3:train dense_v2:train
# choose the score threshold on valid, then test once:
python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split v3:valid --score-thrs 0.2 0.3 0.4 0.5
python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split v3:test --score-thrs <chosen>
python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split v3:test_real --score-thrs <chosen>
python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split dense_v2:test --score-thrs <chosen>   # dense photos (L2)
```
Time (measured in the smoke tests on the shared GPUs, batch 4: r50 0.22 s/it, mamba 0.31 s/it, vmunet 0.39 s/it): at batch 8 and 60k it, expect very roughly **r50 ≈ 7 h, mamba ≈ 10 h, vmunet ≈ 13 h** (not measured end to end; `--max-iter 40000` cuts a third).

Outputs per run: `best.pth`, `model_*.pth`, `last_checkpoint`, `config.yaml` (detectron2), `silk_config.yaml`, `logs/eval_log.csv` (per-larva scores at each evaluation), `metrics.json` (losses).

## Environment
detectron2 (git, commit 247691a) was built into `.venv` on 2026-10-28 with `CUDA_HOME=/usr/local/cuda-12.6` (the `nvcc` 11.3 on PATH is too old for torch cu124). Only new packages were added; torch 2.4.1, numpy and mamba_ssm unchanged. AdelaiDet's own CUDA ops are **not** built (they use the removed THC API); `adet_shim.py` puts an empty module in their place because SOLOv2 never calls them.

## Fairness / risks
- **One change at a time**: compare `mamba` and `vmunet` with `r50` (same heads, data, recipe). But `r50` is ImageNet-pretrained and the VSSM variants are not → also run `r50 --no-imagenet` if the comparison must isolate the backbone architecture.
- Training data by default = `v3:train` (same as E1–E3). Adding `dense_v2:train` is a separate experiment (data change).
- The `p_any` loss for unknown-disease larvae is our addition, not from the paper.
- ⚠️ **Licence: AdelaiDet is for non-commercial use only** (`models/AdelaiDet/LICENSE`; commercial licence from the authors). Fine for the thesis; a production line needs the authors' permission (same situation as SpatialEmbeddings, CC BY-NC).

## Results table
Not trained yet. After training + evaluation: save the scores in `Reports/Meeting_<date>/results/solov2/scores/<variant>/` (`eval_*.txt` + `run.yaml` with `run`, `data: v3`, `score_thr`), run `python utils/export_results_xlsx.py`, fill `Mã` / `Đổi gì` / `Ghi chú`.
