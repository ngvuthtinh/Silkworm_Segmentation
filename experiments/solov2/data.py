"""
data.py — Đăng ký data per-larva của repo (images/, masks_inst/, masks_cls/, masks_ignore/) thành dataset detectron2.

Mỗi con trong masks_inst → một annotation (hộp + mask RLE + lớp):
  masks_cls 1 (Grasserie) → 0, 2 (Healthy) → 1, 3 (chưa biết bệnh) → UNKNOWN (-1)   [xem model.py]
Mỗi mảnh liền >= min_area của masks_ignore → annotation lớp IGNORE (-2): không có loss phân loại ở đó.
Danh sách annotation được lưu đệm ở runs/solov2_cache/ (đọc 20k ảnh mất vài phút).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from experiments.solov2.model import IGNORE, UNKNOWN

DATA_ROOTS = {"v3": "data/sam3_aug20k_v3", "dense_v2": "data/dense_seg_v2"}
CLS_TO_ID = {1: 0, 2: 1, 3: UNKNOWN}
CLASS_NAMES = ["Grasserie", "Healthy"]
CACHE = Path("runs/solov2_cache")


def _rle(m: np.ndarray) -> dict:
    r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def _box(m: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def image_record(root: Path, split: str, img_p: Path) -> dict | None:
    stem = img_p.stem
    inst = cv2.imread(str(root / split / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    cls = cv2.imread(str(root / split / "masks_cls" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if inst is None or cls is None:
        return None
    ign_p = root / split / "masks_ignore" / f"{stem}.png"
    ign = cv2.imread(str(ign_p), cv2.IMREAD_GRAYSCALE) > 127 if ign_p.exists() else np.zeros(inst.shape, bool)
    h, w = inst.shape
    min_area = max(1, round(100 / (256 * 256) * h * w))
    annos = []
    for i in np.unique(inst):
        if i == 0:
            continue
        m = inst == i
        vals, counts = np.unique(cls[m], return_counts=True)
        c = int(vals[np.argmax(counts)])
        if c not in CLS_TO_ID:
            continue
        annos.append({"bbox": _box(m), "bbox_mode": BoxMode.XYXY_ABS, "segmentation": _rle(m), "category_id": CLS_TO_ID[c], "iscrowd": 0})
    if ign.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ign.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= min_area:
                m = lab == k
                annos.append({"bbox": _box(m), "bbox_mode": BoxMode.XYXY_ABS, "segmentation": _rle(m), "category_id": IGNORE, "iscrowd": 0})
    return {"file_name": str(img_p), "image_id": stem, "height": h, "width": w, "annotations": annos,
            "silk_root": str(root), "silk_split": split}


def load_dicts(key: str, split: str, limit: int | None = None) -> list[dict]:
    root = Path(DATA_ROOTS[key])
    cache = CACHE / f"{key}_{split}.pkl"
    if cache.exists() and cache.stat().st_mtime > (root / split / "masks_inst").stat().st_mtime:
        dicts = pickle.loads(cache.read_bytes())
    else:
        imgs = sorted(p for p in (root / split / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        dicts = [r for r in (image_record(root, split, p) for p in imgs) if r is not None]
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(dicts))
    return dicts[:limit] if limit else dicts


def register(spec: str, limit: int | None = None) -> str:
    """spec = '<tên data>:<split>[*<lặp>]' (vd 'v3:train', 'dense_v2:train*8') → tên dataset detectron2 đã đăng ký.
    `*N` lặp danh sách ảnh N lần = trọng số lấy mẫu (nguồn nhỏ như ảnh dày chỉ có 563 ảnh trên 19,989 của v3)."""
    spec, _, rep = spec.partition("*")
    repeat = int(rep or 1)
    key, split = spec.split(":")
    name = f"silk_{key}_{split}" + (f"_x{repeat}" if repeat > 1 else "") + (f"_first{limit}" if limit else "")
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, lambda k=key, s=split, n=limit, r=repeat: load_dicts(k, s, n) * r)
        # không đặt `thing_classes`: detectron2 sẽ in thống kê lớp và báo lỗi với mã UNKNOWN/IGNORE âm
        MetadataCatalog.get(name).set(silk_classes=CLASS_NAMES, silk_root=DATA_ROOTS[key], silk_split=split)
    return name
