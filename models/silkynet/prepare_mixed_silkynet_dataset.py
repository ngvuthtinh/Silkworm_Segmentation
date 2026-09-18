"""
prepare_mixed_silkynet_dataset.py

Mixes single-silkworm samples from SAM3 segmented dataset into the silkworm dataset.
Preserves original filenames (e.g. Healthy-..., Grasserie-...) so disease information is retained.
Supports images, segmentation masks, and YOLO polygon/bbox label files.
Strictly respects split boundaries (train/valid/test) to prevent data leakage.
"""

import os
import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare mixed silkworm dataset with preserved filenames.")
    parser.add_argument("--silkynet-root", type=str, default="methods/silkynet/data",
                        help="Path to legacy silkynet data folder")
    parser.add_argument("--sam3-root", type=str, default="data/Silkworm_SAM3_Segmented",
                        help="Path to SAM3 segmented data folder")
    parser.add_argument("--yolo-root", type=str, default="data/Silkworm Diseases.v1i.yolo26",
                        help="Path to YOLO dataset containing original images")
    parser.add_argument("--output-dir", type=str, default="data/silkworm_mixed_dataset",
                        help="Target output directory for the mixed silkworm dataset")
    parser.add_argument("--num-train-single", type=int, default=500,
                        help="Number of SAM3 single-silkworm images to sample for train split (-1 for all)")
    parser.add_argument("--num-val-single", type=int, default=100,
                        help="Number of SAM3 single-silkworm images to sample for valid split (-1 for all)")
    parser.add_argument("--num-test-single", type=int, default=100,
                        help="Number of SAM3 single-silkworm images to sample for test split (-1 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def load_legacy_silkynet_triplets(silkynet_root: Path):
    """
    Loads (img_path, mask_path, label_path_or_None) from legacy silkynet dataset.
    """
    splits = {'train': [], 'valid': [], 'test': []}

    train_triplets = []
    pairs_sources = [
        (silkynet_root / "larvaTrain" / "img", silkynet_root / "larvaTrain" / "label"),
        (silkynet_root / "output20221127" / "JPEGImages", silkynet_root / "output20221127" / "SegmentationClassPNG")
    ]
    for img_dir, mask_dir in pairs_sources:
        if img_dir.exists() and mask_dir.exists():
            for img_p in sorted(img_dir.iterdir()):
                if img_p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    stem = img_p.stem
                    mask_p = None
                    for ext in [".png", ".jpg", ".jpeg"]:
                        cand = mask_dir / (stem + ext)
                        if cand.exists():
                            mask_p = cand
                            break
                    if mask_p:
                        train_triplets.append((img_p, mask_p, None))
    splits['train'] = train_triplets

    # Valid
    val_img_dir = silkynet_root / "larvaTrain" / "validation_img"
    val_mask_dir = silkynet_root / "larvaTrain" / "validation_label"
    val_triplets = []
    if val_img_dir.exists() and val_mask_dir.exists():
        for img_p in sorted(val_img_dir.iterdir()):
            if img_p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                stem = img_p.stem
                mask_p = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = val_mask_dir / (stem + ext)
                    if cand.exists():
                        mask_p = cand
                        break
                if mask_p:
                    val_triplets.append((img_p, mask_p, None))
    splits['valid'] = val_triplets

    # Test
    test_img_dir = silkynet_root / "larvaTest" / "img"
    test_mask_dir = silkynet_root / "larvaTest" / "label"
    test_triplets = []
    if test_img_dir.exists() and test_mask_dir.exists():
        for img_p in sorted(test_img_dir.iterdir()):
            if img_p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                stem = img_p.stem
                mask_p = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = test_mask_dir / (stem + ext)
                    if cand.exists():
                        mask_p = cand
                        break
                if mask_p:
                    test_triplets.append((img_p, mask_p, None))
    splits['test'] = test_triplets

    return splits


def get_sam3_single_silkworm_triplets(sam3_root: Path, yolo_root: Path, split: str):
    """
    Finds SAM3 samples containing exactly 1 silkworm object.
    Returns list of (img_path, mask_path, label_txt_path).
    """
    lbl_dir = sam3_root / split / "labels"
    mask_dir = sam3_root / split / "masks"
    img_dir = yolo_root / split / "images"

    if not lbl_dir.exists() or not mask_dir.exists() or not img_dir.exists():
        print(f"[WARN] Split {split} directory missing in SAM3 or YOLO root.")
        return []

    single_triplets = []
    for lbl_p in sorted(lbl_dir.glob("*.txt")):
        lines = [line.strip() for line in lbl_p.read_text().splitlines() if line.strip()]
        if len(lines) == 1:
            stem = lbl_p.stem
            mask_p = mask_dir / f"{stem}.png"
            img_candidates = [
                img_dir / f"{stem}.jpg",
                img_dir / f"{stem}.png",
                img_dir / f"{stem}.jpeg"
            ]
            img_p = None
            for cand in img_candidates:
                if cand.exists():
                    img_p = cand
                    break

            if mask_p.exists() and img_p is not None:
                single_triplets.append((img_p, mask_p, lbl_p))

    return single_triplets


def process_triplet(img_src: Path, mask_src: Path, lbl_src: Path | None,
                    img_dst_dir: Path, mask_dst_dir: Path, lbl_dst_dir: Path):
    """
    Copies image, standardizes binary mask, and copies label file keeping ORIGINAL FILENAMES intact.
    """
    # 1. Image (preserve exact original filename)
    img_dst = img_dst_dir / img_src.name
    img = Image.open(img_src).convert("RGB")
    img.save(img_dst)

    # 2. Binary mask (preserve exact original stem with .png)
    mask_dst = mask_dst_dir / f"{img_src.stem}.png"
    mask = Image.open(mask_src).convert("L")
    mask_arr = np.array(mask)
    mask_binary = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
    Image.fromarray(mask_binary).save(mask_dst)

    # 3. Label text file if available (preserve exact original stem with .txt)
    if lbl_src is not None and lbl_src.exists():
        lbl_dst = lbl_dst_dir / f"{img_src.stem}.txt"
        shutil.copy2(lbl_src, lbl_dst)


def main():
    args = parse_args()
    random.seed(args.seed)

    silkynet_root = Path(args.silkynet_root)
    sam3_root = Path(args.sam3_root)
    yolo_root = Path(args.yolo_root)
    output_dir = Path(args.output_dir)

    print("================================================================")
    print(f"Preparing Mixed Silkworm Dataset in: {output_dir}")
    print("Preserving Original Filenames (Healthy / Grasserie retained)")
    print("================================================================")

    legacy_splits = load_legacy_silkynet_triplets(silkynet_root)
    print(f"[INFO] Legacy SilkyNet pairs loaded:")
    print(f"       - Train: {len(legacy_splits['train'])} pairs")
    print(f"       - Valid: {len(legacy_splits['valid'])} pairs")
    print(f"       - Test:  {len(legacy_splits['test'])} pairs")

    num_samples = {
        'train': args.num_train_single,
        'valid': args.num_val_single,
        'test': args.num_test_single
    }

    sam3_selected = {}
    for split in ['train', 'valid', 'test']:
        available = get_sam3_single_silkworm_triplets(sam3_root, yolo_root, split)
        n_req = num_samples[split]
        if n_req < 0 or n_req >= len(available):
            selected = available
        else:
            selected = random.sample(available, n_req)
        sam3_selected[split] = selected
        print(f"[INFO] SAM3 Single Silkworm samples selected for '{split}': {len(selected)} / {len(available)} available")

    summary = {}
    for split in ['train', 'valid', 'test']:
        split_img_dir = output_dir / split / "images"
        split_mask_dir = output_dir / split / "masks"
        split_lbl_dir = output_dir / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        legacy_triplets = legacy_splits[split]
        sam3_triplets = sam3_selected[split]

        total_written = 0
        legacy_count = 0
        sam3_count = 0

        # Process legacy triplets
        for img_p, mask_p, lbl_p in legacy_triplets:
            process_triplet(img_p, mask_p, lbl_p, split_img_dir, split_mask_dir, split_lbl_dir)
            legacy_count += 1
            total_written += 1

        # Process SAM3 triplets (original filenames like Healthy-..., Grasserie-...)
        for img_p, mask_p, lbl_p in sam3_triplets:
            process_triplet(img_p, mask_p, lbl_p, split_img_dir, split_mask_dir, split_lbl_dir)
            sam3_count += 1
            total_written += 1

        summary[split] = {
            "total_samples": total_written,
            "legacy_multi_silkworm": legacy_count,
            "sam3_single_silkworm": sam3_count
        }
        print(f"[SUCCESS] Split '{split}' generated: {total_written} total ({legacy_count} legacy + {sam3_count} single)")

    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n[COMPLETE] Mixed Silkworm Dataset generated successfully at:", output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
