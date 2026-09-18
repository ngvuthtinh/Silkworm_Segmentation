"""
dataset.py — Dataset classes for multi-task silkworm training.

Dataset A (Segmentation) — Silkynet format:
    Structure:
        <root>/
          larvaTrain/
            img/        ← training images (.jpg)
            label/      ← training masks  (.png)
            validation_img/
            validation_label/
          larvaTest/
            img/
            label/
          output20221127/
            JPEGImages/ ← extra images
            SegmentationClassPNG/ ← extra masks

    Item: {"image": [3,H,W], "mask": [1,H,W], "task": "seg"}

Dataset B (Classification) — YOLO format:
    Structure:
        <root>/
          train/images/  + train/labels/
          valid/images/  + valid/labels/
          test/images/   + test/labels/
          data.yaml       (nc, names)

    YOLO class mapping from data.yaml:
        0 = Grasserie (diseased)  → binary label 1
        1 = Healthy               → binary label 0

    Item: {"image": [3,H,W], "class_label": int, "task": "cls"}
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

def _default_train_transform(img_size: int) -> Callable:
    """Standard augmentation pipeline matching original VM-UNet config."""
    return T.Compose([
        T.ToTensor(),                     # [H,W,C] uint8 → [C,H,W] float32 /255
        T.Resize((img_size, img_size), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _default_val_transform(img_size: int) -> Callable:
    return T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _mask_transform(img_size: int) -> Callable:
    """Resize mask and convert to tensor [1, H, W] float in [0, 1]."""
    def _fn(mask_pil: Image.Image) -> torch.Tensor:
        mask_pil = mask_pil.resize((img_size, img_size), Image.NEAREST)
        arr = np.array(mask_pil.convert("L"), dtype=np.float32)
        arr = (arr > 0).astype(np.float32)  # binarise (any non-zero class ID is foreground)
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
    return _fn


# ---------------------------------------------------------------------------
# Dataset A — Segmentation (Silkynet)
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """
    Reads image + binary mask pairs from the Silkynet data layout.

    Supports combining multiple (img_dir, mask_dir) pairs so you can
    merge larvaTrain, output20221127, etc. into a single dataset.

    Parameters
    ----------
    pairs : list of (img_dir, mask_dir) tuples
        Each tuple must point to matching image and mask directories.
    img_size : int
        Spatial size images are resized to.
    train : bool
        If True apply augmentation; otherwise only resize + normalise.
    img_exts : set[str]
        Accepted image file extensions.
    mask_exts : set[str]
        Accepted mask file extensions (tried in order to match image stem).
    """

    IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
    MASK_EXTS = [".png", ".jpg", ".jpeg"]

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        img_size: int = 256,
        train: bool = True,
    ):
        self.img_size  = img_size
        self.train     = train
        self.img_tf    = _default_train_transform(img_size) if train else _default_val_transform(img_size)
        self.mask_tf   = _mask_transform(img_size)
        self.samples: list[tuple[Path, Path]] = []

        for img_dir, mask_dir in pairs:
            img_dir  = Path(img_dir)
            mask_dir = Path(mask_dir)
            if not img_dir.exists():
                raise FileNotFoundError(f"Image dir not found: {img_dir}")
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
            self._collect(img_dir, mask_dir)

        if not self.samples:
            raise RuntimeError("SegDataset: no (image, mask) pairs found.")

    def _collect(self, img_dir: Path, mask_dir: Path) -> None:
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in self.IMG_EXTS:
                continue
            stem = img_path.stem
            mask_path = None
            for ext in self.MASK_EXTS:
                candidate = mask_dir / (stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is None:
                # Try same extension as image
                candidate = mask_dir / img_path.name
                if candidate.exists():
                    mask_path = candidate
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Paired random augmentation (both flip/rotate together)
        if self.train:
            img, mask = self._augment(img, mask)

        img_t  = self.img_tf(img)      # [3, H, W]
        mask_t = self.mask_tf(mask)    # [1, H, W]

        return {
            "image":      img_t,
            "mask":       mask_t,
            "task":       "seg",
            "img_path":   str(img_path),
        }

    @staticmethod
    def _augment(img: Image.Image, mask: Image.Image):
        """Enriched spatially consistent augmentation pipeline for Silkynet images."""
        # 1. Random Resized Crop / Scale & Zoom (simulates zooming into individual silkworms)
        if random.random() < 0.5:
            w, h = img.size
            scale = random.uniform(0.7, 1.0)
            crop_w, crop_h = int(w * scale), int(h * scale)
            i = random.randint(0, h - crop_h)
            j = random.randint(0, w - crop_w)
            img  = TF.crop(img, i, j, crop_h, crop_w)
            mask = TF.crop(mask, i, j, crop_h, crop_w)
            img  = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)

        # 2. Random Flips
        if random.random() < 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. Random Rotations (discrete 90/180/270 or arbitrary -45 to 45 deg)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270, random.uniform(-45, 45)])
            img  = TF.rotate(img,  angle)
            mask = TF.rotate(mask, angle)

        # 4. Random Affine (Shear / Translate)
        if random.random() < 0.3:
            shear = random.uniform(-15, 15)
            dx = random.uniform(-0.05, 0.05) * img.width
            dy = random.uniform(-0.05, 0.05) * img.height
            img  = TF.affine(img, angle=0, translate=[dx, dy], scale=1.0, shear=shear)
            mask = TF.affine(mask, angle=0, translate=[dx, dy], scale=1.0, shear=shear)

        # 5. Image-only Lighting & Color Jitter (does not alter mask geometry)
        if random.random() < 0.6:
            brightness = random.uniform(0.7, 1.3)
            contrast   = random.uniform(0.7, 1.3)
            saturation = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)

        if random.random() < 0.2:
            # Gaussian Blur for camera defocus simulation
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return img, mask

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_silkynet_root(
        cls,
        silkynet_data_dir: str,
        split: str = "train",
        img_size: int = 256,
        include_output20221127: bool = True,
    ) -> "SegDataset":
        """
        Convenience constructor that builds the correct (img_dir, mask_dir)
        pairs from the Silkynet data layout automatically.

        Parameters
        ----------
        silkynet_data_dir : str
            Path to `models/silkynet/data/`
        split : "train" | "val" | "test"
        include_output20221127 : bool
            If True and split=="train", also include the output20221127 folder.
        """
        root = Path(silkynet_data_dir)
        pairs: list[tuple[str, str]] = []
        is_train = split == "train"

        # Check if dataset has standard train/valid/test layout with images/ and masks/
        split_dir_name = "valid" if split in ["val", "valid"] else split
        standard_img_dir = root / split_dir_name / "images"
        standard_mask_dir = root / split_dir_name / "masks"

        if standard_img_dir.exists() and standard_mask_dir.exists():
            pairs.append((str(standard_img_dir), str(standard_mask_dir)))
        elif split == "train":
            pairs.append((
                str(root / "larvaTrain" / "img"),
                str(root / "larvaTrain" / "label"),
            ))
            if include_output20221127:
                pairs.append((
                    str(root / "output20221127" / "JPEGImages"),
                    str(root / "output20221127" / "SegmentationClassPNG"),
                ))
        elif split in ["val", "valid"]:
            pairs.append((
                str(root / "larvaTrain" / "validation_img"),
                str(root / "larvaTrain" / "validation_label"),
            ))
        elif split == "test":
            pairs.append((
                str(root / "larvaTest" / "img"),
                str(root / "larvaTest" / "label"),
            ))
        else:
            raise ValueError(f"Unknown split '{split}'. Choose train/val/test.")

        return cls(pairs=pairs, img_size=img_size, train=is_train)


# ---------------------------------------------------------------------------
# Dataset B — Classification (YOLO)
# ---------------------------------------------------------------------------

class ClsDataset(Dataset):
    """
    Reads images + YOLO bbox labels and converts to image-level binary
    classification labels (0=healthy, 1=diseased).

    YOLO class mapping (from data.yaml):
        class 0 = Grasserie → diseased → binary label 1
        class 1 = Healthy   → healthy  → binary label 0

    For images with multiple boxes (mixed classes), the image-level label
    is 1 (diseased) if ANY box is diseased, else 0 (healthy).

    Parameters
    ----------
    yolo_data_dir : str
        Root of the YOLO dataset (contains data.yaml, train/, valid/, test/).
    split : "train" | "valid" | "test"
    img_size : int
    train : bool
    use_bbox_crop : bool
        If True, crop each bbox region and treat as a separate sample.
        Useful for fine-grained classification with more samples.
    """

    # Map from YOLO class id → binary label (0=healthy, 1=diseased)
    # Based on data.yaml: names: ['Grasserie', 'Healthy']
    _DEFAULT_YOLO2BINARY = {0: 1, 1: 0}   # Grasserie→diseased, Healthy→healthy

    def __init__(
        self,
        yolo_data_dir: str,
        split: str = "train",
        img_size: int = 256,
        train: bool = True,
        use_bbox_crop: bool = False,
        yolo2binary: dict[int, int] | None = None,
        max_samples: int | None = None,
    ):
        self.img_size     = img_size
        self.train        = train
        self.use_bbox_crop = use_bbox_crop
        self.img_tf       = _default_train_transform(img_size) if train else _default_val_transform(img_size)
        self.yolo2binary  = yolo2binary or self._DEFAULT_YOLO2BINARY

        root = Path(yolo_data_dir)
        split_dir  = root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")

        self.samples: list[dict] = []
        self._build_samples(images_dir, labels_dir, use_bbox_crop)

        if max_samples is not None and max_samples > 0:
            self._subsample(max_samples)

        if not self.samples:
            raise RuntimeError(f"ClsDataset: no samples found in {images_dir}")

    def _subsample(self, max_samples: int) -> None:
        """Balanced subsampling across binary classes."""
        if max_samples >= len(self.samples):
            return
        by_class: dict[int, list[dict]] = {}
        for s in self.samples:
            by_class.setdefault(s["class_label"], []).append(s)

        per_class = max_samples // len(by_class)
        subsampled: list[dict] = []
        for lbl, items in sorted(by_class.items()):
            step = max(1, len(items) // per_class)
            subsampled.extend(items[::step][:per_class])

        self.samples = subsampled

    # ------------------------------------------------------------------

    def _build_samples(
        self,
        images_dir: Path,
        labels_dir: Path,
        use_bbox_crop: bool,
    ) -> None:
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            boxes, binary_labels = self._read_label(label_path)

            if not binary_labels:
                # No annotation → skip (image has no silkworm bbox)
                continue

            if use_bbox_crop:
                # One sample per bbox
                for box, lbl in zip(boxes, binary_labels):
                    self.samples.append({
                        "img_path":    img_path,
                        "class_label": lbl,
                        "bbox":        box,   # (cx, cy, w, h) normalised
                    })
            else:
                # One sample per image: label = 1 if ANY box is diseased
                img_label = int(any(l == 1 for l in binary_labels))
                self.samples.append({
                    "img_path":    img_path,
                    "class_label": img_label,
                    "bbox":        None,
                })

    def _read_label(
        self,
        label_path: Path,
    ) -> tuple[list[tuple], list[int]]:
        """Parse YOLO .txt → list of (cx,cy,w,h) and binary labels."""
        if not label_path.exists():
            return [], []
        boxes, binary_labels = [], []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            binary_lbl = self.yolo2binary.get(cls_id, 1)  # unknown class → diseased
            boxes.append((cx, cy, bw, bh))
            binary_labels.append(binary_lbl)
        return boxes, binary_labels

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample    = self.samples[idx]
        img_path  = sample["img_path"]
        cls_label = sample["class_label"]
        bbox      = sample["bbox"]

        img = Image.open(img_path).convert("RGB")

        # Optional bbox crop
        if bbox is not None and self.use_bbox_crop:
            img = self._crop_bbox(img, bbox)

        img_t = self.img_tf(img)  # [3, H, W]

        return {
            "image":       img_t,
            "class_label": torch.tensor(cls_label, dtype=torch.long),
            "task":        "cls",
            "img_path":    str(img_path),
        }

    @staticmethod
    def _crop_bbox(img: Image.Image, bbox: tuple) -> Image.Image:
        """Crop image to YOLO bbox (cx, cy, w, h) normalised → PIL crop."""
        W, H = img.size
        cx, cy, bw, bh = bbox
        x1 = max(0, int((cx - bw / 2) * W))
        y1 = max(0, int((cy - bh / 2) * H))
        x2 = min(W, int((cx + bw / 2) * W))
        y2 = min(H, int((cy + bh / 2) * H))
        if x2 <= x1 or y2 <= y1:
            return img
        return img.crop((x1, y1, x2, y2))

    # ------------------------------------------------------------------
    # Class balance info
    # ------------------------------------------------------------------

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self.samples:
            lbl = s["class_label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Convenience DataLoader factories
# ---------------------------------------------------------------------------

def build_seg_loaders(
    silkynet_data_dir: str,
    img_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for Dataset A."""
    train_ds = SegDataset.from_silkynet_root(silkynet_data_dir, split="train", img_size=img_size)
    val_ds   = SegDataset.from_silkynet_root(silkynet_data_dir, split="val",   img_size=img_size, include_output20221127=False)
    test_ds  = SegDataset.from_silkynet_root(silkynet_data_dir, split="test",  img_size=img_size, include_output20221127=False)

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=shuffle,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)


def build_cls_loaders(
    yolo_data_dir: str,
    img_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    use_bbox_crop: bool = False,
    max_train_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) for Dataset B."""
    train_ds = ClsDataset(yolo_data_dir, split="train", img_size=img_size, train=True,  use_bbox_crop=use_bbox_crop, max_samples=max_train_samples)
    val_ds   = ClsDataset(yolo_data_dir, split="valid", img_size=img_size, train=False, use_bbox_crop=use_bbox_crop)
    test_ds  = ClsDataset(yolo_data_dir, split="test",  img_size=img_size, train=False, use_bbox_crop=use_bbox_crop)

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=shuffle,
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)
