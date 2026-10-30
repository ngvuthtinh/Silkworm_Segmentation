#!/usr/bin/env python3
"""
experiments/multitask_vmunet_disc/inference.py — Vẽ kết quả TỪNG CON: không gộp mảnh (như baseline) và có gộp mảnh bằng embedding.

Mỗi ảnh có nhãn gồm 4 khung: ảnh vào | nhãn thật | dự đoán không gộp | dự đoán có gộp.
Hai khung dự đoán lấy từ CÙNG một lượt chạy model, chỉ khác bước merge_fragments, nên khung 4 so với khung 3
cho thấy riêng tác dụng của embedding.

Một tập có nhãn:
  python -m experiments.multitask_vmunet_disc.inference --checkpoint <run>/checkpoints/best.pth \
      --split test --num 12 --merge-thr 1.6 --output-dir Reports/Meeting_<date>/results/<run>/eval_data_v3/test --gpu 0

Thư mục ảnh mới (không có nhãn):
  python -m experiments.multitask_vmunet_disc.inference --checkpoint ... --image-dir data/test --output-dir ... --gpu 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.multitask_vmunet.inference import _fit, _title
from experiments.multitask_vmunet_disc.config import DiscConfig
from experiments.multitask_vmunet_disc.evaluate import load_cfg_for_checkpoint
from experiments.multitask_vmunet_disc.model import BODY, BOUNDARY, DISEASE, DiscVMUNet
from src.dataset_instance import InstanceSilkwormDataset
from src.dataset_multitask import _normalize
from src.visualize_instances import draw_instances_colored, match_color_index
from src.metrics import diagnose_instances, gt_instance_classes, match_instances, merge_fragments, split_instances


def load_model(ckpt: str, cfg: DiscConfig, device: torch.device) -> DiscVMUNet:
    model = DiscVMUNet(cfg.n_embed, cfg.depths, cfg.depths_decoder, cfg.drop_path_rate).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model.eval()


@torch.no_grad()
def predict(model: DiscVMUNet, rgb: np.ndarray, cfg: DiscConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """rgb [H, W, 3] uint8 → (xác suất [3, S, S] thân/biên/bệnh, embedding [D, S, S])."""
    s = cfg.img_size
    x = _normalize(cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR) / 255.0)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
        logits, emb = model.split(model(t))
    return torch.sigmoid(logits.float())[0].cpu().numpy(), emb.float()[0].cpu().numpy()


def analyse(prob: np.ndarray, emb: np.ndarray, cfg: DiscConfig):
    """Trả về (vùng không gộp, info không gộp, vùng có gộp, info có gộp)."""
    inst0, info0 = split_instances(prob[BODY], prob[BOUNDARY], prob[DISEASE], cfg.body_thr, cfg.boundary_thr, cfg.disease_thr)
    inst1 = merge_fragments(inst0, emb, cfg.merge_thr)
    info1 = diagnose_instances(inst1, prob[DISEASE], cfg.disease_thr)
    return inst0, info0, inst1, info1


def _pred_panel(rgb, inst, info, cfg, gt_inst, gt_cls, ignore, name):
    cnt = match_instances(inst, {k: v[0] for k, v in info.items()}, gt_inst, gt_cls, cfg.iou_thr, cfg.min_area, ignore)
    n_gt, n_pred = cnt["tp"] + cnt["fn"], cnt["tp"] + cnt["fp"]
    _gcol, pcol = match_color_index(inst, gt_inst, cfg.min_area, cfg.iou_thr)       # dự đoán khớp con thật → cùng màu với con đó
    img = draw_instances_colored(rgb, inst, {k: v[0] for k, v in info.items()}, {k: v[1] for k, v in info.items()}, cfg.min_area, pcol)
    color = (120, 255, 120) if cnt["cls_correct"] == n_gt == n_pred else (255, 200, 90)
    return _title(_fit(img), f"{name}: {n_pred} con | tách đúng {cnt['tp']}, chẩn đoán đúng {cnt['cls_correct']}", color), cnt


def render_labeled(rgb, prob, emb, gt_inst, gt_dis, cfg, ignore=None):
    inst0, info0, inst1, info1 = analyse(prob, emb, cfg)
    gt_cls = gt_instance_classes(gt_inst, gt_dis)
    p0, c0 = _pred_panel(rgb, inst0, info0, cfg, gt_inst, gt_cls, ignore, "Không gộp")
    p1, c1 = _pred_panel(rgb, inst1, info1, cfg, gt_inst, gt_cls, ignore, "Có gộp")
    n_gt = c1["tp"] + c1["fn"]
    panels = [
        _title(_fit(rgb), "Ảnh đầu vào"),
        _title(_fit(draw_instances_colored(rgb, gt_inst, gt_cls, None, cfg.min_area, match_color_index(inst1, gt_inst, cfg.min_area, cfg.iou_thr)[0])),
               f"Nhãn thật: {n_gt} con"),
        p0,
        p1,
    ]
    return np.hstack(panels), c0, c1


def render_unlabeled(rgb, prob, emb, cfg) -> np.ndarray:
    inst0, info0, inst1, info1 = analyse(prob, emb, cfg)

    def panel(inst, info, name):
        big = {k: v for k, v in info.items() if (inst == k).sum() >= cfg.min_area}
        img = draw_instances_colored(rgb, inst, {k: v[0] for k, v in big.items()}, {k: v[1] for k, v in big.items()}, cfg.min_area)
        return _title(_fit(img), f"{name}: {len(big)} con")

    return np.hstack([_title(_fit(rgb), "Ảnh đầu vào"), panel(inst0, info0, "Không gộp"), panel(inst1, info1, "Có gộp")])


def main() -> None:
    p = argparse.ArgumentParser(description="Vẽ kết quả từng con của VM-UNet + embedding (không gộp vs có gộp)")
    p.add_argument("--checkpoint", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", help="valid | test | valid_real | test_real (có nhãn thật)")
    g.add_argument("--image-dir", help="thư mục ảnh mới, không có nhãn")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num", type=int, default=12, help="số ảnh vẽ khi dùng --split")
    p.add_argument("--min-larvae", type=int, default=3, help="ưu tiên ảnh có ít nhất ngần này con (khó hơn)")
    p.add_argument("--merge-thr", type=float, default=None, help="ghi đè ngưỡng gộp mảnh")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu", type=str, default="0")
    a = p.parse_args()

    cfg = load_cfg_for_checkpoint(a.checkpoint)
    if a.merge_thr is not None:
        cfg.merge_thr = a.merge_thr
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
        prob, emb = predict(model, rgb, cfg, device)
        vis, c0, c1 = render_labeled(rgb, prob, emb, inst.numpy(), dis[0].numpy(), cfg, valid[0].numpy() < 0.5)
        cv2.imwrite(str(out_dir / f"{k + 1:02d}_{stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        n_gt = c1["tp"] + c1["fn"]
        print(f"  {k + 1:02d} {stem}: nhãn {n_gt} con | không gộp: đúng {c0['tp']}, thừa {c0['fp']} | có gộp: đúng {c1['tp']}, thừa {c1['fp']}")


if __name__ == "__main__":
    main()
