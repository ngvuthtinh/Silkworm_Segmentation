#!/usr/bin/env python3
"""
src/dataset_multitask.py — Unified Multi-Task Dataset

Đọc trực tiếp từ data/Silkworm_mixed_dataset_10k (hoặc bất kỳ dataset nào cùng cấu trúc).
Mỗi mẫu trả về: (image_tensor, binary_mask_tensor, class_label)
  - class_label: 0 = Grasserie (Bệnh), 1 = Healthy (Khỏe)
  - binary_mask: 0 = nền, 1 = thân tằm

Thiết kế để dùng chung cho cả VM-UNet, Swin-UNet và Mask2Former.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Mean/Std ImageNet chuẩn
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _normalize(img: np.ndarray) -> np.ndarray:
    """img: float32 HxWx3 in [0, 1] → normalized HxWx3."""
    img = img.astype(np.float32)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] - _MEAN[c]) / _STD[c]
    return img


class MultitaskSilkwormDataset(Dataset):
    """
    Dataset trả về (image, mask, label) cho Multi-Task Learning.

    Cấu trúc thư mục mong đợi:
        data_root/
          split/
            images/  ← ảnh RGB (.jpg / .png)
            masks/   ← binary mask PNG (0 = bg, 255 = tằm)
            labels/  ← YOLO polygon txt (dòng 1: class_id ...)

    Class mapping (theo YOLO label):
        0 = Grasserie (Bệnh)
        1 = Healthy   (Khỏe)
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "valid", "test"] = "train",
        img_size: int = 256,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")

        split_dir = self.data_root / split
        img_dir   = split_dir / "images"
        mask_dir  = split_dir / "masks"
        lbl_dir   = split_dir / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục images: {img_dir}")

        self.samples: list[tuple[Path, Path, int]] = []

        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = img_p.stem

            # Mask: bắt buộc phải có
            mask_p = mask_dir / f"{stem}.png"
            if not mask_p.exists():
                continue

            # Label: đọc class_id từ dòng đầu tiên YOLO label
            lbl_p = lbl_dir / f"{stem}.txt"
            class_id = self._read_class(lbl_p, stem)

            self.samples.append((img_p, mask_p, class_id))

    @staticmethod
    def _read_class(lbl_p: Path, stem: str) -> int:
        """Đọc class_id từ dòng đầu YOLO label txt. Fallback: suy ra từ tên file."""
        if lbl_p.exists():
            try:
                with open(lbl_p) as f:
                    first = f.readline().strip()
                    if first:
                        return int(first.split()[0])
            except Exception:
                pass
        # Fallback từ tên file
        name_lower = stem.lower()
        if "healthy" in name_lower:
            return 1
        return 0  # mặc định Grasserie

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_p, mask_p, class_id = self.samples[idx]

        # ─── Load ──────────────────────────────────────────────────────────────
        img  = cv2.imread(str(img_p))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # ─── Resize ─────────────────────────────────────────────────────────────
        s = self.img_size
        img  = cv2.resize(img,  (s, s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        # ─── Augment (train only) ────────────────────────────────────────────────
        if self.augment:
            if random.random() < 0.5:
                img  = cv2.flip(img,  1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.5:
                img  = cv2.flip(img,  0)
                mask = cv2.flip(mask, 0)
            k = random.randint(0, 3)
            if k:
                img  = np.rot90(img,  k).copy()
                mask = np.rot90(mask, k).copy()

        # ─── Normalize & to tensor ───────────────────────────────────────────────
        img_f  = _normalize(img / 255.0)                          # HxWx3
        img_t  = torch.from_numpy(img_f.transpose(2, 0, 1))      # 3xHxW
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)  # 1xHxW

        return img_t, mask_t, class_id


# Alias tương thích ngược
JointSilkwormDataset = MultitaskSilkwormDataset

