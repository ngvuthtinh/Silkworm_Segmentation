"""
dataset.py — Dataset classes for Segmentation-First Multi-Task VM-UNet.

Dataset A (Segmentation-only):
    Loads (image, mask) pairs from Silkynet data (larvaTrain & output20221127).

Dataset B (Classification-only):
    Loads (image, class_label) pairs from YOLO Silkworm Diseases dataset.
    Class 0: Healthy | Class 1: Diseased
"""

from __future__ import annotations

import glob
import random
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class SilkynetSegDataset(Dataset):
    """
    Dataset A: Segmentation-only dataset.
    Loads (image, mask) pairs from Silkynet directory structure.
    """

    def __init__(self, data_root: str, img_size: int = 128, is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train
        self.samples: list[tuple[Path, Path]] = []

        root = Path(data_root)

        # Check if root has standard train/images or valid/images layout
        split_name = "train" if is_train else "valid"
        split_img_dir = root / split_name / "images"
        split_mask_dir = root / split_name / "masks"

        if split_img_dir.exists() and split_mask_dir.exists():
            for img_p in sorted(split_img_dir.iterdir()):
                if img_p.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                stem = img_p.stem
                mask_p = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = split_mask_dir / (stem + ext)
                    if cand.exists():
                        mask_p = cand
                        break
                if mask_p:
                    self.samples.append((img_p, mask_p))
        else:
            # Fallback to legacy silkynet layout
            pairs = [
                (root / "larvaTrain" / "img", root / "larvaTrain" / "label"),
                (root / "output20221127" / "JPEGImages", root / "output20221127" / "SegmentationClassPNG"),
            ]

            all_pairs = []
            for img_dir, mask_dir in pairs:
                if not img_dir.exists() or not mask_dir.exists():
                    continue
                for img_p in sorted(img_dir.iterdir()):
                    if img_p.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                        continue
                    stem = img_p.stem
                    mask_p = None
                    for ext in [".png", ".jpg", ".jpeg"]:
                        cand = mask_dir / (stem + ext)
                        if cand.exists():
                            mask_p = cand
                            break
                    if mask_p:
                        all_pairs.append((img_p, mask_p))

            # Split 90% train / 10% val deterministically
            val_count = max(1, int(len(all_pairs) * 0.1))
            if is_train:
                self.samples = all_pairs[:-val_count]
            else:
                self.samples = all_pairs[-val_count:]

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_p, mask_p = self.samples[idx]

        img = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

        # Data Augmentation during training
        if self.is_train:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
                mask = mask.rotate(angle)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))
        img_tensor = self.norm(img_tensor)

        mask_arr = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return img_tensor, mask_tensor


class YOLOClsDataset(Dataset):
    """
    Dataset B: Classification-only dataset.
    Loads (image, class_label) pairs from YOLO dataset directory.
    Class 0: Healthy | Class 1: Diseased
    """

    DISEASE_CLASSES = {
        "muscardine": 1,
        "flacherie": 1,
        "grasserie": 1,
        "pebrine": 1,
        "diseased": 1,
        "healthy": 0,
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 128,
        max_samples: int | None = 500,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.samples: list[tuple[Path, int]] = []

        split_dir = Path(data_root) / split / "images"
        if not split_dir.exists():
            # Fallback search
            candidates = list(Path(data_root).rglob("*.jpg")) + list(Path(data_root).rglob("*.png"))
            for img_p in candidates:
                cls_id = self._parse_class_from_name(img_p.name)
                self.samples.append((img_p, cls_id))
        else:
            for img_p in sorted(split_dir.iterdir()):
                if img_p.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                cls_id = self._parse_class_from_name(img_p.name)
                self.samples.append((img_p, cls_id))

        if max_samples and max_samples < len(self.samples):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _parse_class_from_name(self, filename: str) -> int:
        fname_lower = filename.lower()
        for k, cls_id in self.DISEASE_CLASSES.items():
            if k in fname_lower:
                return cls_id
        return 0  # Default to healthy if unspecified

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_p, label = self.samples[idx]

        img = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))
        img_tensor = self.norm(img_tensor)

        return img_tensor, label
