#!/usr/bin/env python3
"""
experiments/multitask_vmunet/inference.py — Vẽ kết quả TỪNG CON: viền + nhãn Khỏe/Bệnh + xác suất bệnh.

Một tập có nhãn (so với nhãn thật):
  python -m experiments.multitask_vmunet.inference --checkpoint runs/multitask_vmunet/<run>/checkpoints/best.pth \
      --split test --num 12 --output-dir Reports/2026-10-04/test_results/test --gpu 1

Thư mục ảnh mới (không có nhãn):
  python -m experiments.multitask_vmunet.inference --checkpoint ... --image-dir data/test --output-dir ... --gpu 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.multitask_vmunet.config import MultiTaskConfig
from experiments.multitask_vmunet.model import BODY, BOUNDARY, DISEASE, MultiTaskVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.dataset_multitask import _normalize
from src.metrics import gt_instance_classes, match_instances, split_instances

# class_id: 0 = Grasserie (bệnh), 1 = Healthy (khỏe)
COLOR = {0: (255, 90, 30), 1: (60, 200, 60)}
NAME = {0: "Bệnh", 1: "Khỏe"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PANEL = 480


def load_model(ckpt: str, cfg: MultiTaskConfig, device: torch.device) -> MultiTaskVMUNet:
    model = MultiTaskVMUNet(cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()


@torch.no_grad()
def predict(model: MultiTaskVMUNet, rgb: np.ndarray, cfg: MultiTaskConfig, device: torch.device) -> np.ndarray:
    """rgb [H, W, 3] uint8 → xác suất [3, S, S] (thân, biên, bệnh) ở kích thước huấn luyện S."""
    s = cfg.img_size
    x = _normalize(cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR) / 255.0)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
        logits = model(t)
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def _fit(rgb: np.ndarray, size: int = PANEL) -> np.ndarray:
    """Co ảnh vào khung size×size, giữ tỉ lệ, nền xám đậm."""
    h, w = rgb.shape[:2]
    sc = size / max(h, w)
    r = cv2.resize(rgb, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 24, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def draw_instances(
    rgb: np.ndarray,
    inst: np.ndarray,
    classes: dict[int, int],
    scores: dict[int, float] | None = None,
    min_area: int = 100,
) -> np.ndarray:
    """Tô và viền từng con theo lớp, ghi 'Bệnh 93%' / 'Khỏe 7%' (xác suất bệnh) lên từng con."""
    h, w = rgb.shape[:2]
    up = cv2.resize(inst.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    out = rgb.astype(np.float32).copy()
    labels = []
    for i, c in classes.items():
        if (inst == i).sum() < min_area:
            continue
        m = up == i
        if not m.any():
            continue
        col = np.array(COLOR[c], np.float32)
        out[m] = out[m] * 0.6 + col * 0.4
        ys, xs = np.nonzero(m)
        labels.append((int(xs.mean()), int(ys.mean()), c, i, m))
    out = out.clip(0, 255).astype(np.uint8)
    for *_, c, i, m in labels:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, COLOR[c], max(2, w // 300))
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, max(12, w // 32))
    for cx, cy, c, i, _ in labels:
        txt = NAME[c] if scores is None else f"{NAME[c]} {scores[i] * 100:.0f}%"
        d.text((cx, cy), txt, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), anchor="mm")
    return np.array(img)


def _title(panel: np.ndarray, text: str, color: tuple[int, int, int] = (235, 235, 235)) -> np.ndarray:
    bar = Image.new("RGB", (panel.shape[1], 34), (30, 20, 20))
    ImageDraw.Draw(bar).text((10, 6), text, font=ImageFont.truetype(FONT, 17), fill=color)
    return np.vstack([np.array(bar), panel])


def analyse(prob: np.ndarray, cfg: MultiTaskConfig) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
    return split_instances(prob[BODY], prob[BOUNDARY], prob[DISEASE], cfg.body_thr, cfg.boundary_thr, cfg.disease_thr)


def render_labeled(rgb, prob, gt_inst, gt_dis, cfg) -> tuple[np.ndarray, dict[str, int]]:
    pred_inst, info = analyse(prob, cfg)
    pred_cls = {k: v[0] for k, v in info.items()}
    gt_cls = gt_instance_classes(gt_inst, gt_dis)
    cnt = match_instances(pred_inst, pred_cls, gt_inst, gt_cls, cfg.iou_thr, cfg.min_area)
    n_gt = cnt["tp"] + cnt["fn"]
    n_pred = cnt["tp"] + cnt["fp"]
    panels = [
        _title(_fit(rgb), "Ảnh đầu vào"),
        _title(_fit(draw_instances(rgb, gt_inst, gt_cls, None, cfg.min_area)), f"Nhãn thật: {n_gt} con"),
        _title(_fit(draw_instances(rgb, pred_inst, pred_cls, {k: v[1] for k, v in info.items()}, cfg.min_area)),
               f"Dự đoán: {n_pred} con | tách đúng {cnt['tp']}, chẩn đoán đúng {cnt['cls_correct']}",
               (120, 255, 120) if cnt["cls_correct"] == n_gt == n_pred else (255, 200, 90)),
    ]
    return np.hstack(panels), cnt


def render_unlabeled(rgb, prob, cfg) -> np.ndarray:
    pred_inst, info = analyse(prob, cfg)
    big = {k: v for k, v in info.items() if (pred_inst == k).sum() >= cfg.min_area}
    panels = [
        _title(_fit(rgb), "Ảnh đầu vào"),
        _title(_fit(draw_instances(rgb, pred_inst, {k: v[0] for k, v in big.items()},
                                   {k: v[1] for k, v in big.items()}, cfg.min_area)), f"Dự đoán: {len(big)} con"),
    ]
    return np.hstack(panels)


def main() -> None:
    p = argparse.ArgumentParser(description="Vẽ kết quả từng con của VM-UNet multitask")
    p.add_argument("--checkpoint", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", help="valid | test | test_real (có nhãn thật)")
    g.add_argument("--image-dir", help="thư mục ảnh mới, không có nhãn")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num", type=int, default=12, help="số ảnh vẽ khi dùng --split")
    p.add_argument("--min-larvae", type=int, default=3, help="ưu tiên ảnh có ít nhất ngần này con (khó hơn)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu", type=str, default="0")
    a = p.parse_args()

    cfg = MultiTaskConfig()
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(a.checkpoint, cfg, device)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.image_dir:
        for ip in sorted(Path(a.image_dir).iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rgb = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            vis = render_unlabeled(rgb, predict(model, rgb, cfg, device), cfg)
            dst = out_dir / f"{ip.stem.replace(' ', '_')}.png"
            cv2.imwrite(str(dst), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"  {ip.name} → {dst}")
        return

    ds = InstanceSilkwormDataset(cfg.data_root, a.split, cfg.img_size, augment=False, boundary_width=cfg.boundary_width)
    rng = random.Random(a.seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    # ưu tiên ảnh có nhiều con (cảnh khó); phần còn lại lấp bằng ảnh ngẫu nhiên
    hard = [i for i in order if len(np.unique(ds[i][4].numpy())) - 1 >= a.min_larvae]
    chosen = (hard + [i for i in order if i not in set(hard)])[: a.num]

    total: dict[str, int] = {}
    for k, idx in enumerate(chosen):
        _, _, _, dis, inst = ds[idx]
        stem = ds.samples[idx][0].stem
        rgb = cv2.cvtColor(cv2.imread(str(ds.samples[idx][0])), cv2.COLOR_BGR2RGB)
        vis, cnt = render_labeled(rgb, predict(model, rgb, cfg, device), inst.numpy(), dis[0].numpy(), cfg)
        for key, v in cnt.items():
            total[key] = total.get(key, 0) + v
        cv2.imwrite(str(out_dir / f"{k + 1:02d}_{stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  {k + 1:02d} {stem}: nhãn thật {cnt['tp'] + cnt['fn']} con | tách đúng {cnt['tp']} | chẩn đoán đúng {cnt['cls_correct']}")
    n_gt = total["tp"] + total["fn"]
    print(f"Tổng {len(chosen)} ảnh: {n_gt} con thật | tách đúng {total['tp']} | chẩn đoán đúng {total['cls_correct']}")


if __name__ == "__main__":
    main()
