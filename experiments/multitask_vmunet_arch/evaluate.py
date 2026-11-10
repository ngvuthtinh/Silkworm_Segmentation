#!/usr/bin/env python3
"""
experiments/multitask_vmunet_arch/evaluate.py — Đánh giá theo TỪNG CON, đúng như `multitask_vmunet_spatialonly` (spatial embeddings, không watershed).
Chỉ khác ở chỗ nạp model/config theo `arch`. `run_eval` và `spatial_instances` dùng lại từ spatialonly.

Chạy:
  python -m experiments.multitask_vmunet_arch.evaluate --checkpoint <run>/checkpoints/best.pth --split test
  python -m experiments.multitask_vmunet_arch.evaluate --checkpoint ... --split valid --seed-thr 0.9
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from experiments.multitask_vmunet_arch.config import ArchConfig
from experiments.multitask_vmunet_arch.model import build_model
from experiments.multitask_vmunet_spatialonly.evaluate import run_eval, spatial_instances  # noqa: F401  (spatial_instances: các script vẽ ảnh dùng)
from src.dataset_instance import InstanceSilkwormDataset


def load_cfg_for_checkpoint(checkpoint: str) -> ArchConfig:
    """Đọc config.yaml của run chứa checkpoint (arch, depths, số kênh σ, ngưỡng...) để khớp lúc train."""
    saved = Path(checkpoint).resolve().parent.parent / "config.yaml"
    data = yaml.safe_load(saved.read_text()) if saved.exists() else {}
    cfg = ArchConfig(arch=data.get("arch", "base"))
    known = {f.name for f in fields(ArchConfig)}
    for k, v in data.items():
        if k in known and k != "work_dir":
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Đánh giá biến thể kiến trúc VM-UNet + Spatial theo từng con")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", help="valid | test | valid_real | test_real")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--img-size", type=int, default=None, help="cỡ vuông khi suy luận (mặc định 256); ảnh dày chạy ở 512")
    p.add_argument("--seed-thr", type=float, default=None, help="Ghi đè ngưỡng seed (chỉnh trên valid, không chỉnh trên test)")
    args = p.parse_args()

    cfg = load_cfg_for_checkpoint(args.checkpoint)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.seed_thr is not None:
        cfg.seed_thr = args.seed_thr
    if args.img_size:
        k = (args.img_size / 256) ** 2
        cfg.img_size, cfg.min_area, cfg.cluster_min_area = args.img_size, round(100 * k), round(30 * k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

    ds = InstanceSilkwormDataset(cfg.data_root, args.split, cfg.img_size, augment=False, boundary_width=cfg.boundary_width)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=cfg.num_workers)
    res = run_eval(model, dl, cfg, device)
    print(f"\n[{args.split}] {len(ds)} ảnh | arch = {cfg.arch} | data_root = {cfg.data_root} | seed_thr = {cfg.seed_thr}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
