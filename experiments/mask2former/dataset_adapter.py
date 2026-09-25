"""
dataset_adapter.py — Detectron2 dataset registration for silkworm_mixed_dataset in Mask2Former.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog

DATASET_ROOT = "data/silkworm_mixed_dataset"
CLASS_NAMES = ["background", "silkworm"]


def ensure_semantic_masks(split: str, data_root: str = DATASET_ROOT) -> Path:
    """
    Ensures that masks with category IDs (0: background, 1: silkworm) exist in masks_id/.
    The raw masks contain 0 (background) and 255 (silkworm).
    Detectron2 expects pixel values to directly match category IDs (0, 1).
    """
    root = Path(data_root) / split
    raw_mask_dir = root / "masks"
    id_mask_dir = root / "masks_id"
    id_mask_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(raw_mask_dir.glob("*.png"))
    for mf in mask_files:
        target = id_mask_dir / mf.name
        if not target.exists():
            mask_arr = np.array(Image.open(mf))
            # Convert 0/255 to 0/1
            mask_id = (mask_arr > 127).astype(np.uint8)
            Image.fromarray(mask_id).save(target)

    return id_mask_dir


def get_silkworm_mixed_dicts(split: str, data_root: str = DATASET_ROOT) -> list[dict]:
    """
    Generates Detectron2 standard dicts for semantic segmentation.
    """
    root = Path(data_root) / split
    img_dir = root / "images"
    id_mask_dir = ensure_semantic_masks(split, data_root)

    dataset_dicts = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in valid_extensions])

    for idx, img_path in enumerate(img_files):
        stem = img_path.stem
        mask_path = id_mask_dir / f"{stem}.png"

        if not mask_path.exists():
            continue

        # Get image size
        with Image.open(img_path) as im:
            w, h = im.size

        record = {
            "file_name": str(img_path.resolve()),
            "sem_seg_file_name": str(mask_path.resolve()),
            "image_id": idx,
            "height": h,
            "width": w,
        }
        dataset_dicts.append(record)

    return dataset_dicts


def register_silkworm_mixed_datasets(data_root: str = DATASET_ROOT) -> None:
    """
    Registers train, valid, and test sets into Detectron2 DatasetCatalog and MetadataCatalog.
    """
    for split in ["train", "valid", "test"]:
        name = f"silkworm_mixed_{split}"
        if name in DatasetCatalog.list():
            continue

        # Register function with closure
        DatasetCatalog.register(name, lambda s=split, r=data_root: get_silkworm_mixed_dicts(s, r))
        MetadataCatalog.get(name).set(
            stuff_classes=CLASS_NAMES,
            stuff_colors=[(0, 0, 0), (0, 255, 0)],
            evaluator_type="sem_seg",
            ignore_label=255,
        )


if __name__ == "__main__":
    register_silkworm_mixed_datasets()
    print("Registered datasets:", [d for d in DatasetCatalog.list() if "silkworm_mixed" in d])
    for s in ["train", "valid", "test"]:
        dicts = get_silkworm_mixed_dicts(s)
        print(f"Split {s}: {len(dicts)} valid records.")
