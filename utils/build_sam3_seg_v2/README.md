# Building `data/sam3_seg_v2` (clean source)

Run these from the project root. The SAM 3 steps need the `sam3` conda env (`/home/subnh5/miniconda3/envs/sam3/bin/python`). `WORK` is any scratch directory.

| # | Step | Script | Output |
|---|---|---|---|
| 1 | Find broken SAM 3 masks (checked per YOLO box: empty, filled box, speckled, spilling outside the box) | `mask_quality.py` → `box_problems_v3` | list of bad masks |
| 2 | Re-run broken masks with **text prompts only** | `utils/sam3_label_from_yolo.py --no-box-fallback --extra-prompts "silkworm larva" caterpillar worm larva --conf-threshold 0.1 --min-match-iou 0.3` | `WORK/rescue_out/<split>/masks` |
| 3 | Detect every larva with SAM 3 (text prompt) | `detect_all.py WORK/sam3_det` | `WORK/sam3_det/<split>/*.npz` |
| 4 | Group images into scenes (pixel NCC ≥ 0.95 + SIFT ≥ 10 inliers, same class only) | `group_scenes.py WORK` | `WORK/scene_groups_th10.json`, … |
| 5 | Scene-based split, leak-free (groups with any ambiguous link stay in train; ≤ 8 images per valid/test scene) | `build_split.py WORK` + manual valid↔test fix | `split_manifest.json` (this folder) |
| 6 | Write `data/sam3_seg_v2` (repaired masks, degenerate boxes dropped, ignore regions) | `build_v2.py WORK data/sam3_seg_v2` | `data/sam3_seg_v2/` |
| 7 | Real-photo test split | `build_test_real.py data/sam3_seg_v2 data/sam3_aug20k 625` | `data/sam3_aug20k/test_real` |

`split_manifest.json` is the authoritative split. Each entry is `{stem, src_split, split, group, cls}`, where `src_split` is the image's folder in `yolo_bbox`.

## Why these thresholds
- **Pixel NCC 0.95**: all 3,989 Roboflow copy pairs (same photo, different exposure) score ≥ 0.943.
- **SIFT inliers**: pairs of the same photo have a median of 278 inliers; random different scenes stay ≤ 17. Among candidate pairs, the class mismatch rate estimates false links: about 3% at 20–30 inliers, 9% at 15–20, 19% at 12–15, 30% at 10–12. Linking only same-class pairs halves the false links, and over-grouping is the safe direction for leakage.
- **Ignore regions**: SAM 3 detections with score ≥ 0.4, area between 300 px and 25% of the image, box IoU < 0.3 with every YOLO box, and < 30% overlap with the labelled mask. Each is dilated by 7 px.

## Verification (2026-10-06)
- 0 valid/test images match a train image (SIFT ≥ 8 same class, or pixel NCC ≥ 0.85). With the old Roboflow split, ~80% matched.
- Every remaining SIFT ≥ 20 match across splits was inspected. All are different larvae on similar newspaper backgrounds with different classes.
- `group_scenes.py` reproduces exactly the 1,135 groups that were used.
