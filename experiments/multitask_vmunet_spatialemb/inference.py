#!/usr/bin/env python3
"""
experiments/multitask_vmunet_spatialemb/inference.py — Vẽ kết quả TỪNG CON, mỗi con một màu riêng (kiểu SAM3).

Mỗi ảnh có nhãn gồm 5 khung: ảnh vào | nhãn thật | watershed (baseline) | spatial embeddings | seed map.
Hai khung dự đoán lấy từ CÙNG một lượt chạy model, chỉ khác bước tách con. Dự đoán nào khớp một con thật (IoU >= 0.5) cùng màu với
con thật đó; vùng thừa có màu riêng. Một id = một con: các mảnh rời của cùng một con luôn cùng một màu.

Một tập có nhãn:
  python -m experiments.multitask_vmunet_spatialemb.inference --checkpoint <run>/checkpoints/best.pth \
      --split test --num 12 --output-dir Reports/Meeting_<date>/results/<run>/eval_data_v3/test --gpu 0
Thư mục ảnh mới (không có nhãn):
  python -m experiments.multitask_vmunet_spatialemb.inference --checkpoint ... --image-dir data/test --output-dir ... --gpu 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.multitask_vmunet.inference import _fit, _title
from experiments.multitask_vmunet_spatialemb.config import SpatialEmbConfig
from experiments.multitask_vmunet_spatialemb.evaluate import load_cfg_for_checkpoint, spatial_instances, watershed_instances
from experiments.multitask_vmunet_spatialemb.model import SpatialEmbVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.dataset_multitask import _normalize
from src.metrics import gt_instance_classes, match_instances
from src.visualize_instances import draw_instances_colored, match_color_index


def load_model(ckpt: str, cfg: SpatialEmbConfig, device: torch.device) -> SpatialEmbVMUNet:
    model = SpatialEmbVMUNet(cfg.n_sigma, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()


@torch.no_grad()
def predict(model: SpatialEmbVMUNet, rgb: np.ndarray, cfg: SpatialEmbConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """rgb [H, W, 3] uint8 → (xác suất [3, S, S] thân/biên/bệnh, đầu ra thô [2 + n + 1, S, S] offset/σ/seed)."""
    s = cfg.img_size
    x = _normalize(cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR) / 255.0)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
        logits, spatial = model.split(model(t))
    return torch.sigmoid(logits.float())[0].cpu().numpy(), spatial.float()[0].cpu().numpy()


def seed_panel(rgb: np.ndarray, sp: np.ndarray, cfg: SpatialEmbConfig) -> np.ndarray:
    seed = 1.0 / (1.0 + np.exp(-sp[2 + cfg.n_sigma]))
    heat = cv2.cvtColor(cv2.applyColorMap((seed * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base = cv2.resize(rgb, (heat.shape[1], heat.shape[0]), interpolation=cv2.INTER_AREA)
    return (0.35 * base + 0.65 * heat).astype(np.uint8)


def analyse(prob: np.ndarray, sp: np.ndarray, cfg: SpatialEmbConfig):
    i_ws, info_ws = watershed_instances(prob, cfg)
    i_sp, info_sp = spatial_instances(prob, sp, cfg)
    return (i_ws, info_ws), (i_sp, info_sp)


def _pred_panel(rgb, inst, info, cfg, gt_inst, gt_cls, ignore, name):
    cls = {k: v[0] for k, v in info.items()}
    sco = {k: v[1] for k, v in info.items()}
    cnt = match_instances(inst, cls, gt_inst, gt_cls, cfg.iou_thr, cfg.min_area, ignore)
    _gcol, pcol = match_color_index(inst, gt_inst, cfg.min_area, cfg.iou_thr)
    img = draw_instances_colored(rgb, inst, cls, sco, cfg.min_area, pcol)
    n_gt, n_pred = cnt["tp"] + cnt["fn"], cnt["tp"] + cnt["fp"]
    color = (120, 255, 120) if cnt["cls_correct"] == n_gt == n_pred else (255, 200, 90)
    return _title(_fit(img), f"{name}: {n_pred} con | tách đúng {cnt['tp']}, chẩn đoán đúng {cnt['cls_correct']}", color), cnt


def render_labeled(rgb, prob, sp, gt_inst, gt_dis, cfg, ignore=None):
    (i_ws, info_ws), (i_sp, info_sp) = analyse(prob, sp, cfg)
    gt_cls = gt_instance_classes(gt_inst, gt_dis)
    gcol, _ = match_color_index(i_sp, gt_inst, cfg.min_area, cfg.iou_thr)
    p_ws, c_ws = _pred_panel(rgb, i_ws, info_ws, cfg, gt_inst, gt_cls, ignore, "Watershed")
    p_sp, c_sp = _pred_panel(rgb, i_sp, info_sp, cfg, gt_inst, gt_cls, ignore, "Spatial emb")
    n_gt = c_sp["tp"] + c_sp["fn"]
    panels = [
        _title(_fit(rgb), "Ảnh đầu vào"),
        _title(_fit(draw_instances_colored(rgb, gt_inst, gt_cls, None, cfg.min_area, gcol)), f"Nhãn thật: {n_gt} con"),
        p_ws,
        p_sp,
        _title(_fit(seed_panel(rgb, sp, cfg)), "Seed map (nóng = gần tâm con)"),
    ]
    return np.hstack(panels), c_ws, c_sp


def render_unlabeled(rgb, prob, sp, cfg) -> np.ndarray:
    (i_ws, info_ws), (i_sp, info_sp) = analyse(prob, sp, cfg)

    def panel(inst, info, name):
        big = {k: v for k, v in info.items() if (inst == k).sum() >= cfg.min_area}
        img = draw_instances_colored(rgb, inst, {k: v[0] for k, v in big.items()}, {k: v[1] for k, v in big.items()}, cfg.min_area)
        return _title(_fit(img), f"{name}: {len(big)} con")

    return np.hstack([_title(_fit(rgb), "Ảnh đầu vào"), panel(i_ws, info_ws, "Watershed"), panel(i_sp, info_sp, "Spatial emb"),
                      _title(_fit(seed_panel(rgb, sp, cfg)), "Seed map")])


def main() -> None:
    p = argparse.ArgumentParser(description="Vẽ kết quả từng con (mỗi con một màu): watershed vs spatial embeddings")
    p.add_argument("--checkpoint", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", help="valid | test | valid_real | test_real (có nhãn thật)")
    g.add_argument("--image-dir", help="thư mục ảnh mới, không có nhãn")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num", type=int, default=12, help="số ảnh vẽ khi dùng --split")
    p.add_argument("--min-larvae", type=int, default=3, help="ưu tiên ảnh có ít nhất ngần này con (khó hơn)")
    p.add_argument("--seed-thr", type=float, default=None, help="ghi đè ngưỡng seed")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu", type=str, default="0")
    a = p.parse_args()

    cfg = load_cfg_for_checkpoint(a.checkpoint)
    if a.seed_thr is not None:
        cfg.seed_thr = a.seed_thr
    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    model = load_model(a.checkpoint, cfg, device)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.image_dir:
        for ip in sorted(Path(a.image_dir).iterdir()):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rgb = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            vis = render_unlabeled(rgb, *predict(model, rgb, cfg, device), cfg)
            dst = out_dir / f"{ip.stem.replace(' ', '_')}.png"
            cv2.imwrite(str(dst), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"  {ip.name} → {dst}")
        return

    ds = InstanceSilkwormDataset(cfg.data_root, a.split, cfg.img_size, augment=False, boundary_width=cfg.boundary_width)
    rng = random.Random(a.seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    hard = [i for i in order if len(np.unique(ds[i][4].numpy())) - 1 >= a.min_larvae]
    hard_set = set(hard)
    chosen = (hard + [i for i in order if i not in hard_set])[: a.num]

    for k, idx in enumerate(chosen):
        _, _, _, dis, inst, valid = ds[idx]
        stem = ds.samples[idx][0].stem
        rgb = cv2.cvtColor(cv2.imread(str(ds.samples[idx][0])), cv2.COLOR_BGR2RGB)
        prob, sp = predict(model, rgb, cfg, device)
        vis, c0, c1 = render_labeled(rgb, prob, sp, inst.numpy(), dis[0].numpy(), cfg, valid[0].numpy() < 0.5)
        cv2.imwrite(str(out_dir / f"{k + 1:02d}_{stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        n_gt = c1["tp"] + c1["fn"]
        print(f"  {k + 1:02d} {stem}: nhãn {n_gt} con | watershed: đúng {c0['tp']}, thừa {c0['fp']} | spatial: đúng {c1['tp']}, thừa {c1['fp']}")


if __name__ == "__main__":
    main()
