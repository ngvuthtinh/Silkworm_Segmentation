"""
src/visualize_instances.py — Vẽ từng con một màu riêng (kiểu SAM3): tô mờ + viền cùng màu + nhãn chẩn đoán.

Quy ước: một id = một con, mọi mảnh rời của con đó dùng chung MỘT màu. Muốn so sánh dự đoán với nhãn thật bằng mắt,
dùng `match_color_index` để dự đoán nào khớp một con thật (IoU >= 0.5) lấy đúng màu của con thật đó.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NAME = {0: "Bệnh", 1: "Khỏe"}          # class_id: 0 = Grasserie, 1 = Healthy


def instance_color(index: int) -> tuple[int, int, int]:
    """Màu thứ `index`: hue nhảy theo tỉ lệ vàng nên các màu liên tiếp luôn cách xa nhau."""
    r, g, b = colorsys.hsv_to_rgb((index * 0.6180339887) % 1.0, 0.78, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def match_color_index(pred: np.ndarray, gt: np.ndarray, min_area: int = 100, iou_thr: float = 0.5) -> tuple[dict[int, int], dict[int, int]]:
    """
    Trả về (chỉ số màu cho con thật, chỉ số màu cho vùng dự đoán). Vùng dự đoán khớp con thật (ghép tham lam theo IoU)
    lấy cùng chỉ số màu với con thật đó; vùng không khớp nhận chỉ số mới nên không trùng màu với con nào.
    """
    gids = [int(g) for g in np.unique(gt) if g > 0 and (gt == g).sum() >= min_area]
    pids = [int(p) for p in np.unique(pred) if p > 0 and (pred == p).sum() >= min_area]
    gcol = {g: k for k, g in enumerate(gids)}
    pairs = []
    for g in gids:
        gm = gt == g
        cand, cnt = np.unique(pred[gm], return_counts=True)
        for p, c in zip(cand, cnt):
            if p > 0 and int(p) in pids:
                union = gm.sum() + (pred == p).sum() - c
                if c / union >= iou_thr:
                    pairs.append((c / union, g, int(p)))
    pairs.sort(reverse=True)
    pcol: dict[int, int] = {}
    used_g: set[int] = set()
    for _, g, p in pairs:
        if g not in used_g and p not in pcol:
            used_g.add(g)
            pcol[p] = gcol[g]
    nxt = len(gids)
    for p in pids:
        if p not in pcol:
            pcol[p] = nxt
            nxt += 1
    return gcol, pcol


def draw_instances_colored(
    rgb: np.ndarray,
    inst: np.ndarray,
    classes: dict[int, int] | None = None,
    scores: dict[int, float] | None = None,
    min_area: int = 100,
    color_index: dict[int, int] | None = None,
) -> np.ndarray:
    """
    Tô từng con một màu riêng (alpha 0.45) + viền cùng màu + nhãn 'Bệnh 93%' / 'Khỏe 7%' (xác suất bệnh) ở giữa con.
    inst: [h, w] id từng con (0 = nền); classes/scores: {id: class_id} / {id: p_grasserie} (tuỳ chọn);
    color_index: {id: chỉ số màu}; mặc định đánh số theo thứ tự id.
    """
    h, w = rgb.shape[:2]
    up = cv2.resize(inst.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    ids = [int(i) for i in np.unique(inst) if i > 0 and (inst == i).sum() >= min_area]
    if color_index is None:
        color_index = {i: k for k, i in enumerate(ids)}
    out = rgb.astype(np.float32).copy()
    items = []
    for i in ids:
        m = up == i
        if not m.any():
            continue
        col = instance_color(color_index.get(i, len(color_index) + i))
        out[m] = out[m] * 0.55 + np.array(col, np.float32) * 0.45
        ys, xs = np.nonzero(m)
        items.append((i, col, m, int(xs.mean()), int(ys.mean())))
    out = out.clip(0, 255).astype(np.uint8)
    for _i, col, m, *_ in items:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, col, max(2, w // 260))
    if classes is None:
        return out
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, max(12, w // 32))
    for i, _col, _m, cx, cy in items:
        if i not in classes:
            continue
        txt = NAME[classes[i]] if scores is None else f"{NAME[classes[i]]} {scores[i] * 100:.0f}%"
        d.text((cx, cy), txt, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm")
    return np.array(img)
