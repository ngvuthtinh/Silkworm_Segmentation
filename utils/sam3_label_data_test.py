#!/usr/bin/env python3
"""
Cho SAM 3 quét kỹ 2 ảnh trong data/test (ảnh dày, không nhãn) trên ẢNH GỐC rồi dựng mask từng con làm "ground truth" GIẢ (pseudo-GT).

Cách quét giống utils/sam3_scan_dets.py (5 text prompt, ngưỡng thấp 0.1, ảnh gốc + lật ngang + 4 mảnh phóng to, gộp bằng NMS).
Cách dựng nhãn giống utils/build_dense_seg/apply_sam3_missing.py:
  điểm >= 0.7          → thành một con trong masks_inst (lớp 3 = chưa biết bệnh); chừa khe 2 px giữa các con
  điểm 0.5 – 0.7       → vùng bỏ qua (masks_ignore): không loss, không tính điểm
  điểm < 0.5           → không gán (vẫn lưu phát hiện thô để xem)
  Điều kiện: diện tích >= 100 px ở 256² (tính theo cỡ ảnh), dài/rộng >= 1.5 hoặc chạm biên ảnh, không chồng >= 30% lên con đã nhận.

Ra data/test_sam3_gt/ (cùng bố cục các bộ per-larva, một split `test`):
  images/ (bản sao nguyên gốc), masks/, masks_inst/ (uint16), masks_cls/ (3 = chưa biết bệnh), masks_ignore/, dets/<stem>.npz (phát hiện thô),
  previews/<stem>.png (ảnh gốc | các con được nhận), summary.md (số con theo mức điểm)
⚠️ Đây là nhãn GIẢ do SAM 3 tạo, chưa ai kiểm; con SAM 3 bỏ sót hoặc chấm < 0.5 vẫn không có nhãn. Chỉ dùng để so các model với nhau, không phải đáp án thật.

  /home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_label_data_test.py --gpu-id 1
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models" / "sam3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_recheck import scan  # noqa: E402

ASPECT, OVERLAP = 1.5, 0.3
EXT = {".png", ".jpg", ".jpeg"}


def shape(m: np.ndarray) -> tuple[int, float, bool]:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    ys, xs = np.nonzero(m)
    border = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1)
    return int(m.sum()), float(max(rw, rh) / max(min(rw, rh), 1)), border


def build(masks: np.ndarray, scores: np.ndarray, label_thr: float, ignore_thr: float, gap: int):
    h, w = masks.shape[1:] if len(masks) else (0, 0)
    min_area = max(1, round(100 / (256 * 256) * h * w))
    inst = np.zeros((h, w), np.uint16)
    ign = np.zeros((h, w), bool)
    taken = np.zeros((h, w), bool)
    near = np.zeros((h, w), bool)
    k = 0
    band = {"label": 0, "ignore": 0, "below": 0, "rejected_shape": 0}
    kernel = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    for i in np.argsort(-scores):
        s, m = float(scores[i]), masks[i]
        if s < ignore_thr:
            band["below"] += 1
            continue
        area, aspect, border = shape(m) if m.any() else (0, 0.0, False)
        if area < min_area or not (aspect >= ASPECT or border):
            band["rejected_shape"] += 1
            continue
        if (m & taken).sum() / area >= OVERLAP or (m & ign).sum() / area >= 0.5:
            continue
        if s >= label_thr:
            region = m & ~near & ~taken
            if region.sum() < min_area:
                continue
            k += 1
            inst[region] = k
            taken |= region
            near |= cv2.dilate(region.astype(np.uint8), kernel) > 0
            band["label"] += 1
        else:
            region = m & ~taken
            if region.sum() < min_area:
                continue
            ign |= region
            band["ignore"] += 1
    return inst, ign, band, min_area


def preview(rgb: np.ndarray, inst: np.ndarray, ign: np.ndarray) -> np.ndarray:
    import colorsys
    out = rgb.astype(np.float32).copy()
    for i in range(1, int(inst.max()) + 1):
        r, g, b = colorsys.hsv_to_rgb(((i - 1) * 0.6180339887) % 1.0, 0.78, 1.0)
        m = inst == i
        out[m] = 0.45 * out[m] + 0.55 * np.array([r * 255, g * 255, b * 255])
    out[ign] = 0.5 * out[ign] + 0.5 * np.array([150, 150, 150])
    return out.clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/test")
    ap.add_argument("--out", default="data/test_sam3_gt")
    ap.add_argument("--label-thr", type=float, default=0.7)
    ap.add_argument("--ignore-thr", type=float, default=0.5)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--upscale", type=float, default=1.0, help="phóng ảnh trước khi quét (mặc định 1 = ảnh gốc); mask được thu về cỡ gốc")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--checkpoint", default="models/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--bpe-path", default="models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    a = ap.parse_args()

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    torch.cuda.set_device(a.gpu_id)
    model = build_sam3_image_model(bpe_path=str(ROOT / a.bpe_path), checkpoint_path=str(ROOT / a.checkpoint),
                                   enable_inst_interactivity=False, device="cuda", load_from_HF=False)
    processor = Sam3Processor(model, device="cuda")

    out = Path(a.out) / "test"
    for d in ("images", "masks", "masks_inst", "masks_cls", "masks_ignore", "dets", "previews"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = ["# SAM 3 trên 2 ảnh data/test (ảnh gốc)", "",
            f"Quét: 5 text prompt, ngưỡng {a.thr}, ảnh gốc{'' if a.upscale == 1 else f' phóng ×{a.upscale}'} + lật + 4 mảnh phóng to, NMS. "
            f"Nhãn: điểm ≥ {a.label_thr} → con; {a.ignore_thr}–{a.label_thr} → bỏ qua; thấp hơn → không gán.", "",
            "| Ảnh | Cỡ | Phát hiện thô (sau NMS) | Điểm ≥ 0.7 | Điểm 0.5–0.7 | Điểm < 0.5 | **Con có nhãn** | Vùng bỏ qua | Loại vì hình dạng/diện tích |", "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(Path(a.input).iterdir()):
        if p.suffix.lower() not in EXT:
            continue
        rgb = np.array(Image.open(p).convert("RGB"))
        h, w = rgb.shape[:2]
        src = rgb if a.upscale == 1 else cv2.resize(rgb, None, fx=a.upscale, fy=a.upscale, interpolation=cv2.INTER_CUBIC)
        min_area_scan = max(30, round(100 / (256 * 256) * h * w * a.upscale ** 2))
        masks, scores, votes, swmax = scan(processor, src, a.thr, min_area_scan)
        if a.upscale != 1:
            masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in masks]) if len(masks) else np.zeros((0, h, w), bool)
        stem = p.stem
        np.savez_compressed(out / "dets" / f"{stem}.npz", masks=np.packbits(masks, axis=-1), scores=scores, sw_votes=votes, sw_max=swmax, shape=np.array([h, w]))
        inst, ign, band, min_area = build(masks, scores, a.label_thr, a.ignore_thr, a.gap) if len(masks) else (np.zeros((h, w), np.uint16), np.zeros((h, w), bool), {"label": 0, "ignore": 0, "below": 0, "rejected_shape": 0}, 1)
        shutil.copy(p, out / "images" / p.name)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), ((inst > 0) * 255).astype(np.uint8))
        cv2.imwrite(str(out / "masks_inst" / f"{stem}.png"), inst)
        cv2.imwrite(str(out / "masks_cls" / f"{stem}.png"), (inst > 0).astype(np.uint8) * 3)
        cv2.imwrite(str(out / "masks_ignore" / f"{stem}.png"), (ign * 255).astype(np.uint8))
        pv = preview(rgb, inst, ign)
        pv = cv2.resize(np.hstack([rgb, pv]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / "previews" / f"{stem}.png"), cv2.cvtColor(pv, cv2.COLOR_RGB2BGR))
        n = len(scores)
        rows.append(f"| `{p.name}` | {w}×{h} | {n} | {(scores >= a.label_thr).sum()} | {((scores >= a.ignore_thr) & (scores < a.label_thr)).sum()} | {(scores < a.ignore_thr).sum()} | "
                    f"**{band['label']}** | {band['ignore']} | {band['rejected_shape']} (min_area {min_area}px) |")
        print(rows[-1], flush=True)
    (Path(a.out) / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
