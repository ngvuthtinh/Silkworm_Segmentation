#!/usr/bin/env python3
"""
src/dataset_instance.py — Dataset cho bài toán phân đoạn TỪNG CON + chẩn đoán TỪNG CON.

Mỗi mẫu trả về (image, body, boundary, disease, inst, valid):
  - image    [3, H, W] float32  — ảnh RGB chuẩn hóa ImageNet
  - body     [1, H, W] float32  — 1 = pixel thuộc thân tằm
  - boundary [1, H, W] float32  — 1 = pixel nằm trên viền của một con (giữa 2 con, hoặc giữa con và nền)
  - disease  [1, H, W] float32  — 1 = pixel thuộc con bị Grasserie (chỉ có nghĩa trong vùng body)
  - inst     [H, W]    int64    — id từng con (0 = nền), dùng để đánh giá theo từng con
  - valid    [1, H, W] float32  — 1 = pixel được tính loss/điểm, 0 = "vùng bỏ qua" (con tằm có trong ảnh
                                    nhưng không có nhãn, hoặc mảnh vụn quá nhỏ). Không có file → tính tất cả.

Cấu trúc thư mục mong đợi (sinh bởi utils/augment_mixed_10k.py):
    data_root/<split>/images/*.jpg
    data_root/<split>/masks_inst/*.png   (uint16, id từng con)
    data_root/<split>/masks_cls/*.png    (uint8: 0 = nền, 1 = Grasserie, 2 = Healthy)
    data_root/<split>/masks_ignore/*.png (uint8: 255 = vùng bỏ qua; tuỳ chọn)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset_multitask import _normalize

# Giá trị trong masks_cls
CLS_BG, CLS_GRASSERIE, CLS_HEALTHY = 0, 1, 2


def instance_boundary(inst: np.ndarray, width: int = 1) -> np.ndarray:
    """
    Pixel thuộc thân tằm mà trong lân cận (2*width+1)^2 có id khác nó (con khác hoặc nền).
    Với 2 con chạm nhau, viền nằm ở cả hai phía chỗ tiếp xúc → tách được 2 con.
    """
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    inst32 = inst.astype(np.float32)
    differs = cv2.dilate(inst32, k) != cv2.erode(inst32, k)
    return (differs & (inst > 0)).astype(np.float32)


class InstanceSilkwormDataset(Dataset):
    """Dataset trả về các bản đồ theo pixel cho mô hình multitask phân đoạn từng con."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        img_size: int = 256,
        augment: bool = True,
        boundary_width: int = 1,
    ):
        self.root = Path(data_root) / split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.boundary_width = boundary_width

        img_dir = self.root / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục images: {img_dir}")

        self.samples: list[tuple[Path, Path, Path]] = []
        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            inst_p = self.root / "masks_inst" / f"{img_p.stem}.png"
            cls_p = self.root / "masks_cls" / f"{img_p.stem}.png"
            if inst_p.exists() and cls_p.exists():
                self.samples.append((img_p, inst_p, cls_p))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_p, inst_p, cls_p = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        inst = cv2.imread(str(inst_p), cv2.IMREAD_UNCHANGED).astype(np.int32)
        cls = cv2.imread(str(cls_p), cv2.IMREAD_GRAYSCALE)
        ign_p = self.root / "masks_ignore" / f"{img_p.stem}.png"
        ign = cv2.imread(str(ign_p), cv2.IMREAD_GRAYSCALE) if ign_p.exists() else np.zeros(cls.shape, np.uint8)

        s = self.img_size
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        inst = cv2.resize(inst.astype(np.float32), (s, s), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        cls = cv2.resize(cls, (s, s), interpolation=cv2.INTER_NEAREST)
        ign = cv2.resize(ign, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img, inst, cls, ign = img[:, ::-1], inst[:, ::-1], cls[:, ::-1], ign[:, ::-1]
            if random.random() < 0.5:
                img, inst, cls, ign = img[::-1], inst[::-1], cls[::-1], ign[::-1]
            k = random.randint(0, 3)
            if k:
                img, inst, cls, ign = np.rot90(img, k), np.rot90(inst, k), np.rot90(cls, k), np.rot90(ign, k)
            img, inst, cls, ign = img.copy(), inst.copy(), cls.copy(), ign.copy()

        body = (inst > 0).astype(np.float32)
        boundary = instance_boundary(inst, self.boundary_width)
        disease = (cls == CLS_GRASSERIE).astype(np.float32)
        valid = ((ign <= 127) | (inst > 0)).astype(np.float32)   # con có nhãn luôn được tính

        img_t = torch.from_numpy(_normalize(img / 255.0).transpose(2, 0, 1).copy())
        return (
            img_t,
            torch.from_numpy(body)[None],
            torch.from_numpy(boundary)[None],
            torch.from_numpy(disease)[None],
            torch.from_numpy(inst.astype(np.int64)),
            torch.from_numpy(valid)[None],
        )
