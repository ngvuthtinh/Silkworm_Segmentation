#!/usr/bin/env python3
"""
Đối chiếu nhãn của data/dense_seg_v1 với phát hiện của SAM 3: tìm con tằm SAM 3 thấy nhưng nhãn không có.

Bước 1 (GPU, lâu): quét SAM 3, lưu phát hiện thô cho mỗi ảnh (cùng cách như data v3, utils/sam3_scan_dets.py). Kết quả lưu THEO TÊN ẢNH
(data/legacy/sam3_scan_dense/all/dets/<stem>.npz) nên không phụ thuộc ảnh nằm ở split nào; chạy lại chỉ quét ảnh còn thiếu:
  /home/subnh5/miniconda3/envs/sam3/bin/python utils/sam3_scan_dets.py --data-root data/sam3_scan_dense/_input --split all \
      --out-root data/sam3_scan_dense --gpu-id 0   (_input/all/images chứa liên kết tới các ảnh cần quét; kết quả hiện nằm ở data/legacy/sam3_scan_dense)
Bước 2 (CPU, script này): so phát hiện với nhãn, KHÔNG sửa data, chỉ ghi báo cáo:
  python utils/build_dense_seg/check_missing_with_sam3.py --out Reports/Meeting_2026-10-30/results/label_audit/dense_seg_v1_sam3

Một phát hiện được tính là "THIẾU NHÃN" khi (tiêu chuẩn của fix_seg_v3.py, diện tích tính theo cỡ ảnh):
  - điểm SAM 3 >= ngưỡng; diện tích >= min_area (≈ 625 px ở 640²); dài (tỉ lệ dài/rộng >= 1.5) hoặc chạm biên ảnh;
  - < 30% diện tích nằm trong con đã có nhãn, và < 50% nằm trong vùng bỏ qua.
Báo cáo hai mức điểm: >= 0.5 và >= 0.7 (v3 thêm vào nhãn nếu >= 0.7, vào vùng bỏ qua nếu 0.5-0.7).
Đây là gợi ý: SAM 3 cũng có thể nhận nhầm lá, bóng hoặc ghép hai con làm một; xem ảnh trong panels/ để xác nhận bằng mắt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ASPECT, LABEL_OVERLAP, IGNORE_OVERLAP = 1.5, 0.3, 0.5
THRS = (0.5, 0.7)
P, BAR = 520, 40


def shape(m: np.ndarray) -> tuple[int, float, bool]:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    ys, xs = np.nonzero(m)
    border = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1)
    return int(m.sum()), float(max(rw, rh) / max(min(rw, rh), 1)), border


def panel(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    sc = P / max(h, w)
    r = cv2.resize(img, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
    body = np.full((P, P, 3), 255, np.uint8)
    y0, x0 = (P - r.shape[0]) // 2, (P - r.shape[1]) // 2
    body[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    bar = Image.new("RGB", (P, BAR), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    font = ImageFont.truetype(FONT, 20)
    d.text(((P - d.textlength(label, font=font)) / 2, 8), label, font=font, fill=(20, 20, 20))
    return np.vstack([body, np.array(bar)])


def tint(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    out = out.clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/legacy/dense_seg_v1")
    ap.add_argument("--dets-dir", default="data/legacy/sam3_scan_dense/all/dets")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", type=int, default=24, help="số ảnh nghi ngờ nhất được vẽ")
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    manifest = {x["stem"]: x for x in json.loads((root / "manifest.json").read_text())}

    cand_rows, per_image = [], []
    for sp in a.splits:
        for stem in sorted(s for s, x in manifest.items() if x["split"] == sp):
            npz = Path(a.dets_dir) / f"{stem}.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            h, w = (int(v) for v in z["shape"])
            dets = np.unpackbits(z["masks"], axis=-1)[..., :w].astype(bool) if len(z["scores"]) else np.zeros((0, h, w), bool)
            scores = z["scores"]
            inst = cv2.imread(str(root / sp / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            ign = cv2.imread(str(root / sp / "masks_ignore" / f"{stem}.png"), 0) > 127
            lab = inst > 0
            min_area = round(625 / (640 * 640) * h * w)
            n_lab = int(manifest[stem]["larvae_in_label"])
            found = []
            for m, s in zip(dets, scores):
                area, aspect, border = shape(m) if m.any() else (0, 0.0, False)
                if area < min_area or not (aspect >= ASPECT or border):
                    continue
                f_lab, f_ign = float((m & lab).sum() / area), float((m & ign).sum() / area)
                if f_lab >= LABEL_OVERLAP or f_ign >= IGNORE_OVERLAP:
                    continue
                found.append((float(s), area, aspect, border, f_lab, m))
                cand_rows.append({"split": sp, "stem": stem, "score": round(float(s), 3), "area": area, "aspect": round(aspect, 2),
                                  "touches_border": border, "frac_in_label": round(f_lab, 3), "larvae_in_label": n_lab})
            per_image.append({"split": sp, "stem": stem, "n_label": n_lab, "found": found, "path": str(root / sp / "images" / f"{stem}.jpg"),
                              "n05": sum(1 for f in found if f[0] >= 0.5), "n07": sum(1 for f in found if f[0] >= 0.7)})

    lines = ["# SAM 3 đối chiếu với nhãn của `dense_seg_v1`", "",
             "Mỗi con SAM 3 thấy mà nhãn không có và không nằm trong vùng bỏ qua được đếm là **thiếu nhãn** (tiêu chuẩn ở đầu "
             "`check_missing_with_sam3.py`). Train chỉ quét một bản cho mỗi ảnh gốc (3 bản augment cùng cảnh) cộng toàn bộ ảnh Silkynet.", "",
             "| Tập | Ảnh đã quét | Con có nhãn | Con thiếu (điểm ≥ 0.5) | Ảnh có ≥ 1 con thiếu (≥ 0.5) | Con thiếu (điểm ≥ 0.7) | Ảnh có ≥ 1 con thiếu (≥ 0.7) | Thiếu / (có nhãn + thiếu) (≥ 0.5) |",
             "|---|---|---|---|---|---|---|---|"]
    groups = {sp: [r for r in per_image if r["split"] == sp] for sp in a.splits}
    groups["fl (train)"] = [r for r in per_image if r["split"] == "train" and r["stem"].startswith("fl_")]
    groups["silkynet (train)"] = [r for r in per_image if r["stem"].startswith("silkynet_")]
    for name, rs in groups.items():
        if not rs:
            continue
        nl = sum(r["n_label"] for r in rs)
        n5, n7 = sum(r["n05"] for r in rs), sum(r["n07"] for r in rs)
        i5, i7 = sum(1 for r in rs if r["n05"]), sum(1 for r in rs if r["n07"])
        lines.append(f"| {name} | {len(rs)} | {nl} | {n5} | {i5} ({100 * i5 / len(rs):.0f}%) | {n7} | {i7} ({100 * i7 / len(rs):.0f}%) | "
                     f"{100 * n5 / max(nl + n5, 1):.1f}% |")
    lines += ["", "Cách đọc cột cuối: nếu bằng 10%, cứ 10 con thật trong ảnh thì khoảng 1 con chưa có nhãn (theo SAM 3).", ""]

    sus = sorted((r for r in per_image if r["n05"]), key=lambda r: -(sum(f[1] for f in r["found"] if f[0] >= 0.5)))
    for r in sus[:a.panels]:
        rgb = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        sp, stem = r["split"], Path(r["path"]).stem
        inst = cv2.imread(str(root / sp / "masks_inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        miss = np.zeros(inst.shape, bool)
        for f in r["found"]:
            if f[0] >= 0.5:
                miss |= f[5]
        row = np.hstack([panel(rgb, "Ảnh gốc"), np.full((P + BAR, 8, 3), 255, np.uint8),
                         panel(tint(rgb, inst > 0, (60, 170, 60)), "Trước"), np.full((P + BAR, 8, 3), 255, np.uint8),
                         panel(tint(tint(rgb, inst > 0, (60, 170, 60)), miss, (235, 60, 60)), "Sau")])
        cv2.imwrite(str(out / "panels" / f"{sp}__{stem[:60]}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    lines += [f"Ảnh trong `panels/`: {min(len(sus), a.panels)} ảnh có nhiều diện tích thiếu nhất (Trước: xanh = nhãn hiện tại; Sau: thêm đỏ = con SAM 3 tìm thêm, nếu được áp dụng)."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=["split", "stem", "score", "area", "aspect", "touches_border", "frac_in_label", "larvae_in_label"])
        w_.writeheader()
        w_.writerows(cand_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
