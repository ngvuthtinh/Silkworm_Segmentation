"""
dataset_adapter.py — Dataset loader for ViT-P point-based training and evaluation on Silkworm Mixed Dataset.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SilkwormViTPDataset(Dataset):
    """
    Dataset adapter for ViT-P point-based classification.
    Samples point coordinates on foreground (silkworm) and background,
    generating normalized points in [-1, 1] and corresponding binary labels.
    """

    def __init__(
        self,
        split_dir: str,
        input_size: int = 224,
        num_points: int = 32,
        is_train: bool = True,
    ):
        super().__init__()
        self.split_dir = Path(split_dir)
        self.img_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks_id"
        self.input_size = input_size
        self.num_points = num_points
        self.is_train = is_train

        # Fallback to masks/ if masks_id/ doesn't exist
        if not self.mask_dir.exists():
            self.mask_dir = self.split_dir / "masks"

        # List valid paired images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples: List[Tuple[Path, Path]] = []

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() in valid_extensions:
                mask_path = self.mask_dir / f"{img_path.stem}.png"
                if not mask_path.exists():
                    mask_path = self.mask_dir / f"{img_path.stem}.jpg"
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        # Image normalization (standard ImageNet)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize to fixed input size (224x224)
        image = TF.resize(image, [self.input_size, self.input_size], interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.input_size, self.input_size], interpolation=T.InterpolationMode.NEAREST)

        # Random horizontal flip for training
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Convert mask to numpy binary (0 = Background, 1 = Silkworm)
        raw_mask = np.array(mask)
        if raw_mask.max() > 1:
            mask_np = (raw_mask > 127).astype(np.int64)
        else:
            mask_np = (raw_mask > 0).astype(np.int64)
        h, w = mask_np.shape

        # Sample points and labels
        fg_coords = np.argwhere(mask_np == 1)
        bg_coords = np.argwhere(mask_np == 0)

        points = np.zeros((self.num_points, 2), dtype=np.float32)
        labels = np.zeros((self.num_points,), dtype=np.int64)

        half = self.num_points // 2

        # Half foreground, half background (or balanced if foreground exists)
        if len(fg_coords) > 0 and len(bg_coords) > 0:
            fg_idx = np.random.choice(len(fg_coords), size=half, replace=(len(fg_coords) < half))
            bg_idx = np.random.choice(len(bg_coords), size=self.num_points - half, replace=(len(bg_coords) < (self.num_points - half)))

            sampled_fg = fg_coords[fg_idx]
            sampled_bg = bg_coords[bg_idx]

            # Points format: (row/h, col/w) -> normalized [-1, 1]
            points[:half, 0] = (sampled_fg[:, 0] / float(h)) * 2.0 - 1.0
            points[:half, 1] = (sampled_fg[:, 1] / float(w)) * 2.0 - 1.0
            labels[:half] = 1

            points[half:, 0] = (sampled_bg[:, 0] / float(h)) * 2.0 - 1.0
            points[half:, 1] = (sampled_bg[:, 1] / float(w)) * 2.0 - 1.0
            labels[half:] = 0
        else:
            # All background or uniform random
            rand_r = np.random.randint(0, h, size=self.num_points)
            rand_c = np.random.randint(0, w, size=self.num_points)
            points[:, 0] = (rand_r / float(h)) * 2.0 - 1.0
            points[:, 1] = (rand_c / float(w)) * 2.0 - 1.0
            labels[:] = mask_np[rand_r, rand_c]

        # Convert image to Tensor & normalize
        img_tensor = TF.to_tensor(image)
        img_tensor = self.normalize(img_tensor)

        return {
            "image": img_tensor,
            "points": torch.from_numpy(points).float(),
            "labels": torch.from_numpy(labels).long(),
            "mask": torch.from_numpy(mask_np).long(),
            "img_path": str(img_path),
        }
