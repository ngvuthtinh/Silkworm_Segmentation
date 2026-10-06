"""Kiểm tra chất lượng mask SAM 3 so với khung YOLO (cùng tiêu chí dùng để đếm mask hỏng)."""
from __future__ import annotations
import cv2, numpy as np
from pathlib import Path

def yolo_box_mask(label_file: Path, h: int, w: int) -> np.ndarray:
    b = np.zeros((h, w), bool)
    for ln in (label_file.read_text().splitlines() if label_file.exists() else []):
        t = ln.split()
        if len(t) == 5:
            _, xc, yc, bw, bh = map(float, t)
            b[max(int((yc - bh / 2) * h), 0):int((yc + bh / 2) * h), max(int((xc - bw / 2) * w), 0):int((xc + bw / 2) * w)] = True
    return b

def mask_problems(mask_file: Path, label_file: Path) -> list[str]:
    m = cv2.imread(str(mask_file), 0)
    if m is None:
        return ["không có file mask"]
    m = m > 127
    if not m.any():
        return ["mask rỗng"]
    b = yolo_box_mask(label_file, *m.shape)
    _, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    out = []
    if (m & b).sum() / m.sum() < 0.8:
        out.append("nằm ngoài khung")
    if st[1:, 4].max() / m.sum() < 0.9:
        out.append("vỡ nhiều mảnh")
    if m.mean() < 0.002:
        out.append("quá nhỏ")
    return out


def yolo_boxes(label_file: Path, h: int, w: int, margin: float = 0.08) -> list[tuple[int, int, int, int]]:
    out = []
    for ln in (label_file.read_text().splitlines() if label_file.exists() else []):
        t = ln.split()
        if len(t) == 5:
            _, xc, yc, bw, bh = map(float, t)
            mx, my = bw * margin, bh * margin
            out.append((max(int((xc - bw / 2 - mx) * w), 0), max(int((yc - bh / 2 - my) * h), 0),
                        min(int((xc + bw / 2 + mx) * w), w), min(int((yc + bh / 2 + my) * h), h)))
    return out


def box_problems(mask_file: Path, label_file: Path) -> list[str]:
    """Chấm từng khung YOLO: trong khung phải có MỘT con rõ ràng (không rỗng, không tô kín khung, không lốm đốm)."""
    m = cv2.imread(str(mask_file), 0)
    if m is None:
        return ["không có file mask"]
    m = m > 127
    h, w = m.shape
    boxes = yolo_boxes(label_file, h, w)
    if not boxes:
        return ["không có khung"]
    out = set()
    for x1, y1, x2, y2 in boxes:
        r = m[y1:y2, x1:x2]
        fill = r.mean() if r.size else 0
        if fill < 0.05:
            out.add("khung rỗng/gần rỗng")
            continue
        if fill > 0.85:
            out.add("tô kín khung")
        n, _, st, _ = cv2.connectedComponentsWithStats(r.astype(np.uint8))
        if st[1:, 4].max() / r.sum() < 0.8:
            out.add("lốm đốm/nhiều mảnh trong khung")
    covered = np.zeros_like(m)
    for x1, y1, x2, y2 in boxes:
        covered[y1:y2, x1:x2] = True
    if (m & ~covered).sum() / max(m.sum(), 1) > 0.2:
        out.add("tràn ngoài khung")
    return sorted(out)


def box_problems_v3(mask_file: Path, label_file: Path) -> tuple[list[str], int]:
    """
    Trả về (lỗi mask, số khung YOLO lỗi). Chấm từng khung (không nới), bỏ qua khung quá bé (lỗi gán nhãn):
      - rỗng       : mảnh lớn nhất < 3% diện tích khung
      - tô kín     : mask phủ > 85% khung (dấu hiệu nhánh gợi ý-khung chạy trọng số ngẫu nhiên)
      - lốm đốm    : ≥ 8 mảnh tí hon (< 0.5% khung) trong khung
      - tràn       : > 20% pixel mask nằm ngoài mọi khung (đã nới 8%)
    """
    m = cv2.imread(str(mask_file), 0)
    if m is None:
        return ["không có file mask"], 0
    m = m > 127
    h, w = m.shape
    bad_boxes, out = 0, set()
    tight = yolo_boxes(label_file, h, w, margin=0.0)
    loose = yolo_boxes(label_file, h, w, margin=0.08)
    kept = []
    for (x1, y1, x2, y2), lb in zip(tight, loose):
        if (x2 - x1) < 8 or (y2 - y1) < 8 or (x2 - x1) * (y2 - y1) < 0.0005 * h * w:
            bad_boxes += 1
            continue
        kept.append(lb)
        r = m[y1:y2, x1:x2]
        area = r.size
        n, _, st, _ = cv2.connectedComponentsWithStats(r.astype(np.uint8))
        comps = st[1:, 4]
        if comps.size == 0 or comps.max() < 0.03 * area:
            out.add("rỗng")
            continue
        if r.mean() > 0.85:
            out.add("tô kín khung")
        if (comps < 0.005 * area).sum() >= 8:
            out.add("lốm đốm")
    if not kept:
        out.add("không có khung hợp lệ")
    covered = np.zeros_like(m)
    for x1, y1, x2, y2 in kept:
        covered[y1:y2, x1:x2] = True
    if m.any() and (m & ~covered).sum() / m.sum() > 0.2:
        out.add("tràn ngoài khung")
    return sorted(out), bad_boxes
