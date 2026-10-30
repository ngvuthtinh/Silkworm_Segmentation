#!/usr/bin/env python3
"""
experiments/multitask_vmunet_spatialemb/evaluate.py — Đánh giá theo TỪNG CON, hai đường hậu xử lý từ CÙNG một lượt dự đoán:
  - watershed (baseline): split_instances = hạt giống (thân − biên) → watershed; một vùng cho mỗi thành phần liền kề → các khoá *_watershed
  - spatial embeddings  : cluster_spatial_embeddings = chọn seed → gom pixel theo vùng gaussian quanh tâm; không cần liền kề → các khoá chính
Hai kết quả chung model nên khác nhau chỉ ở bước tách con.

Chạy:
  python -m experiments.multitask_vmunet_spatialemb.evaluate --checkpoint <run>/checkpoints/best.pth --split test
  python -m experiments.multitask_vmunet_spatialemb.evaluate --checkpoint ... --split valid --seed-thr 0.4   # chỉnh ngưỡng seed trên valid
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.multitask_vmunet_spatialemb.config import SpatialEmbConfig
from experiments.multitask_vmunet_spatialemb.model import BODY, BOUNDARY, DISEASE, SpatialEmbVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.metrics import (
    cluster_spatial_embeddings,
    diagnose_instances,
    gt_instance_classes,
    match_instances,
    split_instances,
    summarize_instance_counts,
)


def spatial_instances(prob: np.ndarray, sp: np.ndarray, cfg: SpatialEmbConfig) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
    """prob [3, H, W] (thân, biên, bệnh); sp [2 + n + 1, H, W] (offset, σ thô, seed logit) → (bản đồ id, {id: (lớp, p_bệnh)})."""
    n = cfg.n_sigma
    seed = 1.0 / (1.0 + np.exp(-sp[2 + n]))
    inst = cluster_spatial_embeddings(sp[0:2], sp[2:2 + n], seed, prob[BODY] >= cfg.body_thr, cfg.seed_thr, cfg.cluster_min_area,
                                    coord_scale=256)   # cỡ pixel của lúc train (cửa sổ 256²), đúng với mọi cỡ ảnh suy luận
    return inst, diagnose_instances(inst, prob[DISEASE], cfg.disease_thr)


def watershed_instances(prob: np.ndarray, cfg: SpatialEmbConfig) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
    return split_instances(prob[BODY], prob[BOUNDARY], prob[DISEASE], cfg.body_thr, cfg.boundary_thr, cfg.disease_thr)


@torch.no_grad()
def run_eval(
    model: SpatialEmbVMUNet,
    loader: DataLoader,
    cfg: SpatialEmbConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Dice thân tằm + các chỉ số theo từng con, cho đường spatial embeddings (khoá chính) và đường watershed (*_watershed)."""
    model.eval()
    sp_counts: collections.Counter = collections.Counter()
    ws_counts: collections.Counter = collections.Counter()
    n_sp = n_ws = 0
    inter = denom = 0.0
    for b, (imgs, body, _bnd, disease, inst, valid) in enumerate(tqdm(loader, desc="  → [VAL]  ", ncols=130, leave=False)):
        if max_batches is not None and b >= max_batches:
            break
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits, spatial = model.split(model(imgs.to(device)))
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        spatial = spatial.float().cpu().numpy()
        pb = prob[:, BODY] >= cfg.body_thr
        gb = body[:, 0].numpy() > 0.5
        vm = valid[:, 0].numpy() > 0.5
        inter += (pb & gb & vm).sum()
        denom += (pb & vm).sum() + (gb & vm).sum()
        for i in range(prob.shape[0]):
            gi, gd = inst[i].numpy(), disease[i, 0].numpy()
            gcls = gt_instance_classes(gi, gd)
            i_ws, info_ws = watershed_instances(prob[i], cfg)
            i_sp, info_sp = spatial_instances(prob[i], spatial[i], cfg)
            n_ws += len(info_ws)
            n_sp += len(info_sp)
            ws_counts.update(match_instances(i_ws, {k: v[0] for k, v in info_ws.items()}, gi, gcls, cfg.iou_thr, cfg.min_area, ~vm[i]))
            sp_counts.update(match_instances(i_sp, {k: v[0] for k, v in info_sp.items()}, gi, gcls, cfg.iou_thr, cfg.min_area, ~vm[i]))
    out = summarize_instance_counts(sp_counts)
    ws = summarize_instance_counts(ws_counts)
    for k in ("inst_precision", "inst_recall", "inst_f1", "cls_acc_matched", "e2e_f1", "grasserie_recall"):
        out[f"{k}_watershed"] = ws[k]
    out["body_dice"] = float(2 * inter / max(denom, 1))
    out["n_gt_worms"] = int(sp_counts["tp"] + sp_counts["fn"])
    out["n_regions_spatial"] = int(n_sp)
    out["n_regions_watershed"] = int(n_ws)
    return out


def load_cfg_for_checkpoint(checkpoint: str) -> SpatialEmbConfig:
    """Đọc config.yaml của run chứa checkpoint (nếu có) để số kênh σ, ngưỡng... khớp lúc train."""
    cfg = SpatialEmbConfig()
    saved = Path(checkpoint).resolve().parent.parent / "config.yaml"
    if saved.exists():
        known = {f.name for f in fields(SpatialEmbConfig)}
        for k, v in yaml.safe_load(saved.read_text()).items():
            if k in known and k != "work_dir":
                setattr(cfg, k, v)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Đánh giá VM-UNet + spatial embeddings theo từng con")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", help="valid | test | valid_real | test_real")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--data-root", type=str, default=None, help="Ghi đè thư mục data trong config")
    p.add_argument("--img-size", type=int, default=None, help="cỡ vuông khi suy luận (mặc định = lúc train, 256); ảnh dày chạy ở 512. min_area tính theo diện tích tương đối")
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
    model = SpatialEmbVMUNet(cfg.n_sigma, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

    ds = InstanceSilkwormDataset(cfg.data_root, args.split, cfg.img_size, augment=False, boundary_width=cfg.boundary_width)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=cfg.num_workers)
    res = run_eval(model, dl, cfg, device)
    print(f"\n[{args.split}] {len(ds)} ảnh | data_root = {cfg.data_root} | seed_thr = {cfg.seed_thr}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
