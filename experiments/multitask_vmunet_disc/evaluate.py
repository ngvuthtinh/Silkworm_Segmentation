#!/usr/bin/env python3
"""
experiments/multitask_vmunet_disc/evaluate.py — Đánh giá theo TỪNG CON, có và không gộp mảnh bằng embedding.

Mỗi ảnh được xử lý hai lần từ CÙNG một lượt dự đoán:
  - không gộp : split_instances như baseline (một vùng cho mỗi thành phần liền kề)  → các khoá *_nomerge
  - có gộp    : thêm merge_fragments (gộp các mảnh rời có embedding gần nhau)       → các khoá chính (e2e_f1, ...)
Hiệu hai kết quả cho biết riêng bước gộp mảnh đóng góp bao nhiêu.

Chạy:
  python -m experiments.multitask_vmunet_disc.evaluate --checkpoint runs/multitask_vmunet_disc/<run>/checkpoints/best.pth
  python -m experiments.multitask_vmunet_disc.evaluate --checkpoint ... --split valid --merge-thr 1.2   # chỉnh ngưỡng gộp trên valid
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import fields
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.multitask_vmunet_disc.config import DiscConfig
from experiments.multitask_vmunet_disc.model import BODY, BOUNDARY, DISEASE, DiscVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.metrics import (
    diagnose_instances,
    gt_instance_classes,
    match_instances,
    merge_fragments,
    split_instances,
    summarize_instance_counts,
)


@torch.no_grad()
def run_eval(
    model: DiscVMUNet,
    loader: DataLoader,
    cfg: DiscConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Dice thân tằm + các chỉ số theo từng con (tách đúng, chẩn đoán đúng), có và không gộp mảnh."""
    model.eval()
    merged_counts: collections.Counter = collections.Counter()
    plain_counts: collections.Counter = collections.Counter()
    n_before = n_after = 0
    inter = denom = 0.0
    for b, (imgs, body, _bnd, disease, inst, valid) in enumerate(tqdm(loader, desc="  → [VAL]  ", ncols=130, leave=False)):
        if max_batches is not None and b >= max_batches:
            break
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits, emb = model.split(model(imgs.to(device)))
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        emb = emb.float().cpu().numpy()
        pb = prob[:, BODY] >= cfg.body_thr
        gb = body[:, 0].numpy() > 0.5
        vm = valid[:, 0].numpy() > 0.5                      # chỉ tính ngoài vùng bỏ qua
        inter += (pb & gb & vm).sum()
        denom += (pb & vm).sum() + (gb & vm).sum()
        for i in range(prob.shape[0]):
            inst0, info0 = split_instances(
                prob[i, BODY], prob[i, BOUNDARY], prob[i, DISEASE], cfg.body_thr, cfg.boundary_thr, cfg.disease_thr
            )
            inst1 = merge_fragments(inst0, emb[i], cfg.merge_thr)
            info1 = diagnose_instances(inst1, prob[i, DISEASE], cfg.disease_thr)
            n_before += len(info0)
            n_after += len(info1)
            gi, gd = inst[i].numpy(), disease[i, 0].numpy()
            gcls = gt_instance_classes(gi, gd)
            plain_counts.update(match_instances(
                inst0, {k: v[0] for k, v in info0.items()}, gi, gcls, cfg.iou_thr, cfg.min_area, ~vm[i]))
            merged_counts.update(match_instances(
                inst1, {k: v[0] for k, v in info1.items()}, gi, gcls, cfg.iou_thr, cfg.min_area, ~vm[i]))
    out = summarize_instance_counts(merged_counts)
    plain = summarize_instance_counts(plain_counts)
    for k in ("inst_precision", "inst_recall", "inst_f1", "e2e_f1"):
        out[f"{k}_nomerge"] = plain[k]
    out["body_dice"] = float(2 * inter / max(denom, 1))
    out["n_gt_worms"] = int(merged_counts["tp"] + merged_counts["fn"])
    out["n_regions_before_merge"] = int(n_before)
    out["n_regions_after_merge"] = int(n_after)
    return out


def load_cfg_for_checkpoint(checkpoint: str) -> DiscConfig:
    """Đọc config.yaml của run chứa checkpoint (nếu có) để số chiều embedding và δ khớp lúc train."""
    cfg = DiscConfig()
    saved = Path(checkpoint).resolve().parent.parent / "config.yaml"
    if saved.exists():
        known = {f.name for f in fields(DiscConfig)}
        for k, v in yaml.safe_load(saved.read_text()).items():
            if k in known and k != "work_dir":
                setattr(cfg, k, v)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Đánh giá VM-UNet + embedding theo từng con")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", help="valid | test | valid_real | test_real")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--data-root", type=str, default=None, help="Ghi đè thư mục data trong config")
    p.add_argument("--merge-thr", type=float, default=None, help="Ghi đè ngưỡng gộp mảnh (chỉnh trên valid, không chỉnh trên test)")
    args = p.parse_args()

    cfg = load_cfg_for_checkpoint(args.checkpoint)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.merge_thr is not None:
        cfg.merge_thr = args.merge_thr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiscVMUNet(cfg.n_embed, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

    ds = InstanceSilkwormDataset(cfg.data_root, args.split, cfg.img_size, augment=False, boundary_width=cfg.boundary_width)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=cfg.num_workers)
    res = run_eval(model, dl, cfg, device)
    print(f"\n[{args.split}] {len(ds)} ảnh | data_root = {cfg.data_root} | merge_thr = {cfg.merge_thr}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
