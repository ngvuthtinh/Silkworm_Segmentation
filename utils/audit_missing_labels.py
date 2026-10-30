#!/usr/bin/env python3
"""
Tìm con tằm THIẾU nhãn trong data per-larva: model đã train thấy một con nhưng mask đáp án không có nó.

  PYTHONPATH=. python utils/audit_missing_labels.py --checkpoint <run SpatialEmb>/checkpoints/best.pth \
      --splits valid_real test_real valid test --out Reports/Meeting_2026-10-30/results/label_audit --gpu 1

Mỗi vùng model dự đoán (>= min_area px ở 256², đường spatial embeddings) được xếp vào một loại:
  khớp nhãn     >= 50% vùng nằm trong con có nhãn (kể cả khi tách/gộp sai: lỗi của model, không phải của nhãn)
  vùng bỏ qua   >= 50% vùng nằm trong `masks_ignore/` (đã biết là thiếu nhãn; không tính vào điểm, không có loss)
  THIẾU NHÃN    < 10% nằm trong nhãn và < 10% trong vùng bỏ qua → con thật chưa có nhãn cũng chưa được đánh dấu bỏ qua:
                loại này làm điểm precision bị tính thấp hơn thực tế (con model tìm đúng bị đếm là "thừa")
  khác          còn lại (nằm lấn một phần), không kết luận
Đây là máy gợi ý chứ không phải sự thật: một vùng "THIẾU NHÃN" có thể là bóng, lá, con quá nhỏ hoặc model sai.
Xem ảnh trong <out>/panels/ để xác nhận bằng mắt; ảnh sắp theo (diện tích × xác suất thân trung bình) giảm dần.

Chỉ ĐỌC data và checkpoint; ghi báo cáo, candidates.csv và panels vào --out.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.multitask_vmunet_spatialemb.evaluate import load_cfg_for_checkpoint, spatial_instances
from experiments.multitask_vmunet_spatialemb.model import BODY, SpatialEmbVMUNet
from src.dataset_instance import InstanceSilkwormDataset

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
P, BAR = 420, 40
SHOW = {"THIẾU NHÃN": (235, 60, 60), "vùng bỏ qua": (150, 150, 150)}


def _panel(img: np.ndarray, label: str) -> np.ndarray:
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


def _tint(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    out = out.clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def _up(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0


@torch.no_grad()
def audit_split(model, cfg, ds: InstanceSilkwormDataset, device: torch.device) -> list[dict]:
    dl = DataLoader(ds, 8, shuffle=False, num_workers=cfg.num_workers)
    rows, k = [], 0
    for imgs, _body, _bnd, _dis, inst, valid in tqdm(dl, desc=f"  {ds.split}", ncols=100, leave=False):
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits, spatial = model.split(model(imgs.to(device)))
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        spatial = spatial.float().cpu().numpy()
        for i in range(prob.shape[0]):
            gt_any = inst[i].numpy() > 0
            ign = valid[i, 0].numpy() < 0.5
            pred, _info = spatial_instances(prob[i], spatial[i], cfg)
            for pid in np.unique(pred):
                if pid == 0:
                    continue
                m = pred == pid
                area = int(m.sum())
                if area < cfg.min_area:
                    continue
                f_gt, f_ign = float((m & gt_any).sum() / area), float((m & ign).sum() / area)
                kind = ("khớp nhãn" if f_gt >= 0.5 else "vùng bỏ qua" if f_ign >= 0.5
                        else "THIẾU NHÃN" if f_gt < 0.1 and f_ign < 0.1 else "khác")
                rows.append({"split": ds.split, "stem": ds.samples[k + i][0].stem, "path": str(ds.samples[k + i][0]),
                             "kind": kind, "area_256": area, "frac_in_label": round(f_gt, 3), "frac_in_ignore": round(f_ign, 3),
                             "mean_body_prob": round(float(prob[i, BODY][m].mean()), 3), "_mask": m, "_gt": gt_any, "_ign": ign})
        k += prob.shape[0]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint SpatialEmb (E2) dùng làm máy gợi ý")
    ap.add_argument("--splits", nargs="+", default=["valid_real", "test_real", "valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels-per-split", type=int, default=12)
    ap.add_argument("--seed-thr", type=float, default=0.9)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    cfg = load_cfg_for_checkpoint(a.checkpoint)
    cfg.seed_thr = a.seed_thr
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    model = SpatialEmbVMUNet(cfg.n_sigma, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu")["model"])
    model.eval()
    out = Path(a.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)

    lines = ["# Kiểm tra con thiếu nhãn", "", f"Máy gợi ý: `{a.checkpoint}` (đường spatial embeddings, seed_thr {a.seed_thr}). "
             "Mỗi con model tìm được được so với nhãn và vùng bỏ qua (xem `utils/audit_missing_labels.py`).", "",
             "| Tập | Ảnh | Ảnh có vùng bỏ qua | Vùng khớp nhãn | Vùng nằm trong vùng bỏ qua | **Vùng THIẾU NHÃN** | **Ảnh có ≥ 1 vùng THIẾU NHÃN** |",
             "|---|---|---|---|---|---|---|"]
    all_rows = []
    for split in a.splits:
        ds = InstanceSilkwormDataset(cfg.data_root, split, cfg.img_size, augment=False, boundary_width=cfg.boundary_width)
        ds.split = split
        rows = audit_split(model, cfg, ds, device)
        n_ign_img = sum(1 for i in range(len(ds)) if (ds[i][5][0].numpy() < 0.5).any())
        cnt = {k: sum(1 for r in rows if r["kind"] == k) for k in ("khớp nhãn", "vùng bỏ qua", "THIẾU NHÃN", "khác")}
        miss_imgs = sorted({r["stem"] for r in rows if r["kind"] == "THIẾU NHÃN"})
        lines.append(f"| `{split}` | {len(ds)} | {n_ign_img} ({100 * n_ign_img / len(ds):.0f}%) | {cnt['khớp nhãn']} | {cnt['vùng bỏ qua']} | "
                     f"**{cnt['THIẾU NHÃN']}** | **{len(miss_imgs)}** ({100 * len(miss_imgs) / len(ds):.1f}%) |")
        # panels: các vùng đáng ngờ nhất (THIẾU NHÃN trước, theo diện tích × xác suất thân)
        sus = sorted((r for r in rows if r["kind"] == "THIẾU NHÃN"), key=lambda r: -r["area_256"] * r["mean_body_prob"])
        seen: set[str] = set()
        for r in sus:
            if r["stem"] in seen or len(seen) >= a.panels_per_split:
                continue
            seen.add(r["stem"])
            rgb = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
            hw = rgb.shape[:2]
            gt_panel = _tint(rgb, _up(r["_gt"], hw), (60, 170, 60)) if r["_gt"].any() else rgb
            if r["_ign"].any():
                gt_panel = _tint(gt_panel, _up(r["_ign"], hw), SHOW["vùng bỏ qua"], 0.5)
            # mọi vùng THIẾU NHÃN của ảnh này
            miss = np.zeros(hw, bool)
            for q in rows:
                if q["stem"] == r["stem"] and q["kind"] == "THIẾU NHÃN":
                    miss |= _up(q["_mask"], hw)
            row = np.hstack([_panel(rgb, "Ảnh gốc"), np.full((P + BAR, 8, 3), 255, np.uint8),
                             _panel(gt_panel, "Nhãn hiện tại"), np.full((P + BAR, 8, 3), 255, np.uint8),
                             _panel(_tint(rgb, miss, SHOW["THIẾU NHÃN"]), "Model tìm thêm")])
            cv2.imwrite(str(out / "panels" / f"{split}__{r['stem'][:60]}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        all_rows += rows
    lines += ["", "Màu trong ảnh `panels/`: xanh lá = nhãn hiện tại, xám = vùng bỏ qua, đỏ = vùng model tìm thêm mà nhãn không có.",
              "Chú ý: số liệu trên là CẬN TRÊN ước lượng (gồm cả báo động giả của model: bóng, lá, con quá nhỏ). Xác nhận bằng mắt ở `panels/`."]
    (out / "missing_labels_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["split", "stem", "kind", "area_256", "frac_in_label", "frac_in_ignore", "mean_body_prob", "path"]
    with open(out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            if r["kind"] in ("THIẾU NHÃN", "vùng bỏ qua"):
                w.writerow({k: r[k] for k in fields})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
