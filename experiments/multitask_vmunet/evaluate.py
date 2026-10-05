#!/usr/bin/env python3
"""
experiments/multitask_vmunet/evaluate.py — Đánh giá theo TỪNG CON.

Chạy:
  python -m experiments.multitask_vmunet.evaluate --checkpoint runs/multitask_vmunet/<run>/checkpoints/best.pth
  python -m experiments.multitask_vmunet.evaluate --checkpoint ... --split test_real
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.multitask_vmunet.config import MultiTaskConfig
from experiments.multitask_vmunet.model import BODY, BOUNDARY, DISEASE, MultiTaskVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.metrics import gt_instance_classes, match_instances, split_instances, summarize_instance_counts


@torch.no_grad()
def run_eval(
    model: MultiTaskVMUNet,
    loader: DataLoader,
    cfg: MultiTaskConfig,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Dice thân tằm (theo pixel) + các chỉ số theo từng con (tách đúng, chẩn đoán đúng)."""
    model.eval()
    counts: collections.Counter = collections.Counter()
    inter = denom = 0.0
    for b, (imgs, body, _bnd, disease, inst) in enumerate(tqdm(loader, desc="  → [VAL]  ", ncols=130, leave=False)):
        if max_batches is not None and b >= max_batches:
            break
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            prob = torch.sigmoid(model(imgs.to(device)).float()).cpu().numpy()
        pb = prob[:, BODY] >= cfg.body_thr
        gb = body[:, 0].numpy() > 0.5
        inter += (pb & gb).sum()
        denom += pb.sum() + gb.sum()
        for i in range(prob.shape[0]):
            pred_inst, info = split_instances(
                prob[i, BODY], prob[i, BOUNDARY], prob[i, DISEASE],
                cfg.body_thr, cfg.boundary_thr, cfg.disease_thr,
            )
            gi, gd = inst[i].numpy(), disease[i, 0].numpy()
            counts.update(match_instances(
                pred_inst, {k: v[0] for k, v in info.items()}, gi, gt_instance_classes(gi, gd),
                cfg.iou_thr, cfg.min_area,
            ))
    out = summarize_instance_counts(counts)
    out["body_dice"] = float(2 * inter / max(denom, 1))
    out["n_gt_worms"] = int(counts["tp"] + counts["fn"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Đánh giá VM-UNet theo từng con")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", help="valid | test | test_real")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    cfg = MultiTaskConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskVMUNet(cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])

    ds = InstanceSilkwormDataset(cfg.data_root, args.split, cfg.img_size, augment=False,
                                 boundary_width=cfg.boundary_width)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=cfg.num_workers)
    res = run_eval(model, dl, cfg, device)
    print(f"\n[{args.split}] {len(ds)} ảnh")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
