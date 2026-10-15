# 📊 Dataset Architecture & Data Handling — Silkworm Segmentation

This document describes every dataset inside `data/`, how each one was built and checked, and how to load the data in [`src/dataset_instance.py`](src/dataset_instance.py).

---

## 📁 1. Folder Layout

Folder names are short and lowercase. The suffix tells you the content: `_bbox` means bounding boxes, `_seg` means segmentation masks, and `_augNk` means about N thousand augmented training images.

```text
data/
├── yolo_bbox/          # Raw data: images + YOLO boxes + disease class (Roboflow export, original split)
├── sam3_aug20k_v3/     # ⭐ Per-larva dataset used for training and testing (train / valid / test / valid_real / test_real)
├── test/               # 2 dense real photos outside the training domain, for qualitative checks
└── legacy/             # Older data, kept to reproduce earlier results (see §2.6)
    ├── sam3_seg_v2/    #   Source masks before the SAM 3 rescan (input of the v3 fix)
    ├── sam3_seg_v3/    #   ⭐ Source masks after the SAM 3 rescan (input of sam3_aug20k_v3)
    ├── sam3_scan/      #   Raw SAM 3 rescan detections (dets/<stem>.npz), input of fix_seg_v3.py
    ├── sam3_aug20k/    #   Per-larva set built from sam3_seg_v2 (masks not yet fixed)
    ├── sam3_aug20k_v1/ #   Leaky split (approximate rebuild)
    ├── sam3_seg/       #   SAM 3 masks v1 (approximate rebuild)
    ├── mixed_10k/      #   Image-level set with Silkynet (approximate rebuild)
    └── silkynet_seg/   #   Silkynet images, no longer used
```

Lineage: `yolo_bbox` → SAM 3 (+ repair, scene split, ignore regions) → `sam3_seg_v2` → SAM 3 rescan + mask fix → `sam3_seg_v3` → augment → `sam3_aug20k_v3`

> 📦 Moved to `data/legacy/` on 2026-10-15: `sam3_aug20k`, `silkynet_seg`, `sam3_scan`, `sam3_seg_v2`, `sam3_seg_v3`. The scripts that rebuild data (`fix_seg_v3.py`, `sam3_scan_dets.py`, `build_aug20k_v3.sh`) still hard-code `data/sam3_seg_v2`, `data/sam3_scan`, `data/sam3_seg_v3`: move the folders back, or edit the paths, before rerunning them. Training and evaluation only read `data/sam3_aug20k_v3`.

---

## 🏷️ 2. Dataset Specifications

Class mapping used everywhere: **`0` = Grasserie (diseased), `1` = Healthy**. The class comes from the filename prefix (`Grasserie*` / `Healthy*`). Every YOLO label line agrees with it (checked on all 8,975 images). The disease labels have not been checked by an expert.

### 2.1. `yolo_bbox`: raw YOLO dataset
- 8,975 images at 640×640. Original Roboflow split: train 7,978, valid 499, test 498.
- Roboflow made 2 copies of every train image, differing only by a random exposure change (±22%). No flips or rotations were applied.
- ⚠️ **The original split leaks.** The photos come in long sessions: many shots of the same larva on the same leaf, sometimes more than 70. The dataset has 8,975 images but only **~1,135 distinct scenes**, and Roboflow split them image by image. SIFT matching found a same-scene image in train for **~80% of valid and test images**. Do not use this split for evaluation.
- About 12–14% of images contain larvae that were never boxed, mostly at the image border or behind the main larva. 9 boxes are degenerate (a few pixels wide, mis-clicks).

### 2.2. `sam3_seg`: SAM 3 pseudo-masks v1
- Same stems as `yolo_bbox`, holding `masks/`, `boundaries/`, `labels/` (no images).
- ⚠️ About 3.2% of masks are broken: speckled noise, an empty box, or a fully filled box. **Root cause:** `sam3.1_multiplex.pt` has **no weights for SAM 3's interactive (box-prompt) predictor**. Whenever the text prompt did not match a box, the script fell back to box prompting, and that branch ran with random weights. Use `--no-box-fallback` (see §4).

### 2.3. `sam3_seg_v2`: clean source ⭐
Built by [`utils/build_sam3_seg_v2/`](utils/build_sam3_seg_v2/README.md):

| Fix | Result |
|---|---|
| Broken SAM 3 masks re-run with text prompts only (`--no-box-fallback`) | 265 of 286 repaired, 21 images dropped |
| Degenerate YOLO boxes | 8 boxes removed |
| **Scene-based split** (pixel + SIFT grouping; a group lives in exactly one split) | 0 valid/test images match any train image (SIFT or pixel) |
| Unlabelled larvae found by SAM 3 (text prompt, score ≥ 0.4, not overlapping a box) | 3,334 larvae → `masks_ignore/` |

| Split | Images | Scenes | Grasserie / Healthy | Images with ignore regions |
|---|---|---|---|---|
| train | 6,906 | 587 | 3,341 / 3,565 | 18% |
| valid | 506 | 252 | 253 / 253 | 27% |
| test | 499 | 296 | 251 / 248 | 22% |

Each valid or test scene contributes at most 8 images, using one Roboflow copy per original. 1,064 near-duplicate images are left out on purpose. Every split contains `images/`, `masks/` (0/255), `masks_ignore/` (255 marks an ignore region), and `labels/` (YOLO boxes). `split_manifest.json` lists every image with its source split, new split and scene group.

### 2.4. `sam3_seg_v3`: source masks fixed by a SAM 3 rescan ⭐
A thorough SAM 3 rescan of every real image (`utils/sam3_scan_dets.py`: 5 text prompts, threshold 0.1, original + horizontal flip + four zoomed tiles, merged by NMS) was compared with `sam3_seg_v2`; `utils/build_sam3_seg_v2/fix_seg_v3.py` then fixed the masks. Rules, per image (a "larva" is a connected blob ≥ 625 px):

1. **Remove bad larvae**: SAM 3 does not see it (IoU < 0.3 with every detection of score ≥ 0.5) **and** the mask is speckled (solidity < 0.5 or perimeter/√area > 14). If an image already has one such larva, every other larva SAM 3 does not see in that image is removed too (garbage masks break into many blobs).
2. **Add missing larvae** (SAM 3 score ≥ 0.5, area ≥ 1000 px, elongated or touching the border, not inside an existing larva or ignore region): score ≥ 0.7 → added to the mask (2 px gap to neighbours); 0.5–0.7 → added to `masks_ignore/`.
3. Images left with no larva are dropped. `labels/` (YOLO boxes) are not copied, they no longer match the masks.

| Split | Images kept | Dropped | Bad larvae removed | Larvae added to mask | Added to ignore | Images changed |
|---|---|---|---|---|---|---|
| train | 6,872 | 34 | 149 (58 images) | 82 | 203 | 293 (4.2%) |
| valid | 505 | 1 | 9 | 9 | 18 | 25 |
| test | 499 | 0 | 11 | 9 | 10 | 17 |

Every change is listed in `<split>/changes.json`. The thresholds are not hand-verified image by image.

### 2.5. `sam3_aug20k_v3`: per-larva dataset ⭐
Synthetic scenes composed from `sam3_seg_v3`. Each scene has several larvae, healthy and diseased mixed, larvae overlapping each other, leaves covering larvae, and photometric changes. The backgrounds and the pasted larvae of each split come only from the same split of `sam3_seg_v3`. `train` is the real photos plus composed scenes up to 20,000 images.

| Split | Images | Larvae | Larvae / image | Images mixing both classes | Images with ignore regions |
|---|---|---|---|---|---|
| `train` | 20,000 (6,872 real + 13,128 composed) | 33,941 | 1.70 | 23% | 31% |
| `valid` | 1,000 | 1,975 | 1.98 | 35% | 44% |
| `test` | 1,000 | 1,916 | 1.92 | 30% | 36% |
| `valid_real` | 505 | 515 | 1.02 | 0% | 29% |
| `test_real` | 499 | 516 | 1.03 | 0% | 24% |

`valid_real` and `test_real` hold the real photos of `sam3_seg_v3/valid` and `/test`, not composed.

> ⚠️ **Counting convention.** One larva = one id in `masks_inst`, **even when a leaf hides it into several disjoint pieces** (about 1.8 pieces per larva on composed images). The post-processing (`split_instances`) returns one region per connected piece, so it cannot output one region for two pieces. Consequences on `test`: perfect maps fed as predictions reach inst_f1 **0.794** (ceiling) when a larva is one id, but **0.990** if every visible piece counts as a larva. The trained model scores e2e_f1 0.743 vs 0.886 under the two conventions. Older documents counted pieces (2.87 larvae/image, ceiling 0.97), which is not the convention `evaluate.py` uses. Real photos are unaffected (every connected blob is one larva by construction, so `test_real` cannot show this effect).

Each split contains:
- `images/`: 640×640 JPG
- `masks/`: binary mask (0/255)
- `masks_inst/`: **uint16 map with one id per larva** (0 = background)
- `masks_cls/`: **uint8 map: 0 = background, 1 = Grasserie, 2 = Healthy**
- `masks_ignore/`: **255 = ignore region**. These pixels carry no loss and no score. They cover unlabelled larvae, larvae SAM 3 is only 0.5–0.7 sure about, and fragments under 625 px at 640² (= 100 px at 256²).
- `labels/`: YOLO polygons, one line per larva
- `aug_previews/`: overlays (gray = ignore region)

Checks run on every split: file sets match; `masks` = `masks_inst > 0`; one class per larva; no larva under 625 px; no labelled larva inside an ignore region; dataset loads.

Post-processing ceiling (ground-truth maps fed in as predictions, separation F1, one id per larva): valid 0.799, test 0.794, valid_real 1.000, test_real 1.000.

Generation (all steps are in `utils/build_sam3_seg_v2/build_aug20k_v3.sh`, which reads `data/sam3_seg_v3`):
```bash
bash utils/build_sam3_seg_v2/build_aug20k_v3.sh   # build_real.py for train/valid_real/test_real, then augment_mixed_10k.py for train (goal 20000 − real images), valid and test
```

Caveats:
- Synthetic scenes are easier than real production images in some ways and not realistic in others (pasted leaves, one background per scene). Always report `test_real` as well.
- `test_real` is 98% single-larva photos, so it measures diagnosis more than separation. A small hand-labelled set of real multi-larva photos is still the missing piece.
- Masks are SAM 3 pseudo-masks and classes come from file names; nothing was checked by an expert. Scores measure agreement with these labels.
- The data has only ~587 distinct real scenes in train; composing more images does not add scene diversity. Photos unlike the training photos (small, dense larvae on leaves) are not handled (see `Reports/Meeting_2026-10-16/results/limitations_summary.md`).

### 2.6. Legacy sets (`data/legacy/`)
- `sam3_seg_v2`: source masks before the rescan; see §2.3. `sam3_scan`: raw rescan detections used by `fix_seg_v3.py`.
- `sam3_aug20k`: per-larva set built from `sam3_seg_v2` (clean scene split, masks not yet fixed; larvae/image 2.87 counted pieces). Models trained on it: `2026-10-14_00-01-58`.
- `sam3_aug20k_v1`: the first per-larva set, built from `sam3_seg` with the leaky Roboflow split and heavy noise (`GaussNoise` default std of 20–44%). Run `2026-10-03_23-12-56` used it, so its scores are optimistic. `sam3_seg`, `sam3_aug20k_v1` and `mixed_10k` are approximate rebuilds of folders deleted on 2026-10-12.
- `mixed_10k`: image-level set. Silkynet masks are empty (value 75 thresholded at 127), Silkynet larvae carry a forced `Healthy` label, 31% of images mix both classes under one label, and the test set leaks.
- `silkynet_seg`: masks use the value 75 instead of 255.

---

## 🔄 3. Loading Data

```python
from src.dataset_instance import InstanceSilkwormDataset
ds = InstanceSilkwormDataset("data/sam3_aug20k_v3", split="train", img_size=256, augment=True)
image, body, boundary, disease, inst, valid = ds[0]
# image    [3,256,256]  ImageNet-normalized
# body     [1,256,256]  1 = larva pixel
# boundary [1,256,256]  1 = pixel on a larva contour
# disease  [1,256,256]  1 = pixel of a Grasserie larva
# inst     [256,256]    larva ids (for per-larva evaluation)
# valid    [1,256,256]  0 = ignore region (no loss, no score); all ones if masks_ignore/ is missing
```
Splits: `train`, `valid`, `test`, `valid_real`, `test_real`. Flip and rotate augmentation applies to `train` only.

---

## 🛠️ 4. Utilities

```bash
# SAM 3 masks from YOLO boxes. With sam3.1_multiplex.pt ALWAYS pass --no-box-fallback
python utils/sam3_label_from_yolo.py --images-dir <imgs> --labels-dir <labels> --output-dir <out> \
    --prompt silkworm --extra-prompts "silkworm larva" caterpillar worm larva \
    --conf-threshold 0.1 --min-match-iou 0.3 --no-box-fallback \
    --checkpoint models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt --bpe-path models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Rebuild the clean source: see utils/build_sam3_seg_v2/README.md

# Thorough SAM 3 rescan of every real image (conda env 'sam3'; resumable; one process per GPU)
/home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_scan_dets.py --split train --shard 0 --num-shards 2 --gpu-id 0
/home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_scan_dets.py --split train --shard 1 --num-shards 2 --gpu-id 1   # → data/sam3_scan/train/dets/

# Fix masks → data/sam3_seg_v3, then rebuild the per-larva dataset → data/sam3_aug20k_v3
python utils/build_sam3_seg_v2/fix_seg_v3.py train valid test
bash utils/build_sam3_seg_v2/build_aug20k_v3.sh

# Review tools: images where SAM 3 and the labels disagree (3 columns: image | labels | SAM 3)
/home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_recheck.py --split test_real --output-dir <out>
/home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_check_on_data.py --split test --num 100 --output-dir <out>   # SAM 3 vs labels, zero-shot disease prompts
```
(`utils/build_sam3_seg_v2/fix_real_labels.py` is an earlier attempt that fixes only the real splits; superseded by `fix_seg_v3.py`.)

Main flags of `augment_mixed_10k.py`: `--source-root`, `--split`, `--goal`, `--paste-scale`, `--paste-counts`, `--min-inst-area`, `--sam3-only`, `--name-prefix`, `--seed`.

---

## 🛡️ 5. Integrity Rules

1. **Raw data is read-only.** Never edit `yolo_bbox/` or the `sam3_seg_*` source folders by hand. Build a new version (`*_v2`, `*_v3`, …) instead and keep the old one in `data/legacy/`.
2. **Split by scene, never by image.** Near-identical shots of one larva must stay in the same split.
3. **Never commit data.** `data/` and `runs/` are excluded by `.gitignore`. Small manifests live in `utils/build_sam3_seg_v2/`; the list of fixes of each split is `data/.../sam3_seg_v3/<split>/changes.json`.
4. **Strict filename matching.** `images/<stem>.jpg` ↔ `masks*/<stem>.png` ↔ `labels/<stem>.txt`.
