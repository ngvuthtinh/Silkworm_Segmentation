# 📊 Dataset Architecture & Data Handling — Silkworm Segmentation

This document describes every dataset inside `data/`, how each one was built and checked, and how to load the data in [`src/dataset_instance.py`](src/dataset_instance.py).

---

## 📁 1. Folder Layout

Folder names are short and lowercase. The suffix tells you the content: `_bbox` means bounding boxes, `_seg` means segmentation masks, and `_augNk` means about N thousand augmented training images.

```text
data/
├── yolo_bbox/        # Raw data: images + YOLO boxes + disease class (Roboflow export, original split)
├── sam3_seg/         # SAM 3 pseudo-masks v1 (contain broken masks, see 2.2)
├── sam3_seg_v2/      # ⭐ Clean source: scene-based split, repaired masks, ignore regions
├── sam3_aug20k/      # ⭐ Per-larva training set built from sam3_seg_v2 (train / valid / test / test_real)
├── sam3_aug20k_v1/   # Previous per-larva set (built from sam3_seg + the Roboflow split; leaky, kept for comparison)
├── mixed_10k/        # Legacy image-level set (SAM3 + Silkynet)
├── silkynet_seg/     # Silkynet images (no disease label), no longer used
└── test/             # 2 unseen real photos for qualitative checks
```

Lineage: `yolo_bbox` → SAM 3 → `sam3_seg` → (repair, scene split, ignore regions) → `sam3_seg_v2` → augment → `sam3_aug20k`

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

### 2.4. `sam3_aug20k`: per-larva training set ⭐
Synthetic scenes composed from `sam3_seg_v2`. Each scene has several larvae, healthy and diseased mixed, larvae overlapping each other, leaves covering larvae, and photometric changes. The backgrounds and the pasted larvae of each split come only from the same split of `sam3_seg_v2`.

| Split | Images | Larvae / image | Images mixing both classes | Images with ignore regions |
|---|---|---|---|---|
| `train` | 20,000 | 2.87 | 36% | 43% |
| `valid` | 1,000 | 2.87 | 34% | 49% |
| `test` | 1,000 | 2.90 | 36% | 44% |
| `test_real` | 499 | 1.04 | 0% | 22% |

`test_real` holds the real photos of `sam3_seg_v2/test`, not composed.

Each split contains:
- `images/`: 640×640 JPG
- `masks/`: binary mask (0/255)
- `masks_inst/`: **uint16 map with one id per larva** (0 = background)
- `masks_cls/`: **uint8 map: 0 = background, 1 = Grasserie, 2 = Healthy**
- `masks_ignore/`: **255 = ignore region**. These pixels carry no loss and no score. They cover unlabelled larvae, and fragments under 625 px at 640² (= 100 px at 256²).
- `labels/`: YOLO polygons, one line per larva
- `aug_previews/`: overlays (gray = ignore region)

Checks run on every split: file sets match; `masks` = `masks_inst > 0`; one class per larva; no larva under 625 px; no labelled larva inside an ignore region; dataset loads.

Post-processing ceiling (ground-truth maps fed in as predictions, separation F1): valid 0.974, test 0.973, test_real 0.976.

Generation:
```bash
OPTS="--sam3-only --source-root data/sam3_seg_v2 --paste-scale 0.6 1.1 --paste-counts 1 2 3 --min-inst-area 625 --workers 24"
python utils/augment_mixed_10k.py $OPTS --name-prefix aug       --split train --output-dir data/sam3_aug20k/train --goal 20000 --seed 42
python utils/augment_mixed_10k.py $OPTS --name-prefix aug_valid --split valid --output-dir data/sam3_aug20k/valid --goal 1000  --seed 101
python utils/augment_mixed_10k.py $OPTS --name-prefix aug_test  --split test  --output-dir data/sam3_aug20k/test  --goal 1000  --seed 101
python utils/build_sam3_seg_v2/build_test_real.py data/sam3_seg_v2 data/sam3_aug20k 625
```

Caveats:
- Synthetic scenes are easier than real production images. Always report `test_real` as well.
- `test_real` is 98% single-larva photos, so it measures diagnosis more than separation. A small hand-labelled set of real multi-larva photos is still the missing piece.

### 2.5. Legacy sets
- `sam3_aug20k_v1`: the previous per-larva set, built from `sam3_seg` with the leaky Roboflow split and heavy noise (`GaussNoise` default std of 20–44%). Kept only to compare with the first trained model.
- `mixed_10k`: image-level set. Silkynet masks are empty (value 75 thresholded at 127), Silkynet larvae carry a forced `Healthy` label, 31% of images mix both classes under one label, and the test set leaks.
- `silkynet_seg`: masks use the value 75 instead of 255.

---

## 🔄 3. Loading Data

```python
from src.dataset_instance import InstanceSilkwormDataset
ds = InstanceSilkwormDataset("data/sam3_aug20k", split="train", img_size=256, augment=True)
image, body, boundary, disease, inst, valid = ds[0]
# image    [3,256,256]  ImageNet-normalized
# body     [1,256,256]  1 = larva pixel
# boundary [1,256,256]  1 = pixel on a larva contour
# disease  [1,256,256]  1 = pixel of a Grasserie larva
# inst     [256,256]    larva ids (for per-larva evaluation)
# valid    [1,256,256]  0 = ignore region (no loss, no score); all ones if masks_ignore/ is missing
```
Splits: `train`, `valid`, `test`, `test_real`. Flip and rotate augmentation applies to `train` only.

---

## 🛠️ 4. Utilities

```bash
# SAM 3 masks from YOLO boxes. With sam3.1_multiplex.pt ALWAYS pass --no-box-fallback
python utils/sam3_label_from_yolo.py --images-dir <imgs> --labels-dir <labels> --output-dir <out> \
    --prompt silkworm --extra-prompts "silkworm larva" caterpillar worm larva \
    --conf-threshold 0.1 --min-match-iou 0.3 --no-box-fallback \
    --checkpoint models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt --bpe-path models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# Rebuild the clean source: see utils/build_sam3_seg_v2/README.md
```
Main flags of `augment_mixed_10k.py`: `--source-root`, `--split`, `--goal`, `--paste-scale`, `--paste-counts`, `--min-inst-area`, `--sam3-only`, `--name-prefix`, `--seed`.

---

## 🛡️ 5. Integrity Rules

1. **Raw data is read-only.** Never edit `yolo_bbox/` or `sam3_seg/` by hand. Build new versions (`*_v2`) instead.
2. **Split by scene, never by image.** Near-identical shots of one larva must stay in the same split.
3. **Never commit data.** `data/` and `runs/` are excluded by `.gitignore`. Small manifests live in `utils/build_sam3_seg_v2/`.
4. **Strict filename matching.** `images/<stem>.jpg` ↔ `masks*/<stem>.png` ↔ `labels/<stem>.txt`.
