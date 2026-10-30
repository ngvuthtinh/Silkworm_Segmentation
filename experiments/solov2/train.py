#!/usr/bin/env python3
"""
experiments/solov2/train.py — Train SOLOv2 (SilkSOLOv2) trên data tằm với một trong ba backbone.

Chạy thử (20 vòng, 2 ảnh/batch, chấm 8 ảnh valid):
  python -m experiments.solov2.train --variant mamba --smoke-test --gpu 1
Chạy thật (trong tmux):
  python -m experiments.solov2.train --variant r50      --gpu 1 --out-dir Reports/Meeting_<date>/checkpoints/solov2_r50
  python -m experiments.solov2.train --variant mamba --gpu 1 --out-dir Reports/Meeting_<date>/checkpoints/solov2_mamba
  python -m experiments.solov2.train --variant vmunet   --gpu 1 --out-dir Reports/Meeting_<date>/checkpoints/solov2_vmunet
Tiếp tục run bị ngắt:  ... --resume <thư mục run>

Ra: <run>/model_*.pth, best.pth (theo e2e_f1 trên valid), last_checkpoint, config.yaml (detectron2), silk_config.yaml,
    logs/eval_log.csv (điểm theo từng con mỗi lần chấm), metrics.json (loss theo vòng, của detectron2).
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, hooks
from detectron2.solver.build import get_default_optimizer_params
from detectron2.utils.logger import setup_logger

from experiments.solov2 import backbones  # noqa: F401  (đăng ký hai backbone VSSM)
from experiments.solov2 import model as _model  # noqa: F401  (đăng ký META_ARCH SilkSOLOv2)
from experiments.solov2.config import SOLOv2Config, build_cfg
from experiments.solov2.data import register
from experiments.solov2.evaluator import SilkEvaluator


def adamw_full_clip(clip: float) -> type:
    """AdamW cắt chuẩn gradient của TOÀN model trước mỗi bước (AMP: GradScaler đã bỏ hệ số nhân trước khi gọi step)."""

    class FullClipAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            torch.nn.utils.clip_grad_norm_([p for g in self.param_groups for p in g["params"]], clip)
            return super().step(closure)

    return FullClipAdamW


class Trainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(model, base_lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                              weight_decay_norm=0.0)
        return adamw_full_clip(cfg.SILK.CLIP_FULL_MODEL)(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    @classmethod
    def build_train_loader(cls, cfg):
        """Scale jitter (cạnh ngắn ngẫu nhiên ≈ 0.6-1.4× cỡ gốc) rồi CẮT cửa sổ crop² ở cỡ gốc, lật ngang/dọc.
        Cắt chỉ để vừa bộ nhớ và cho model thấy nhiều kích thước; lúc chấm / suy luận chạy CẢ ảnh (MIN_SIZE_TEST)."""
        size = tuple(cfg.INPUT.CROP.SIZE)
        augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"),
                T.FixedSizeCrop(size, pad=True, pad_value=0.0, seg_pad_value=0),
                T.RandomFlip(horizontal=True, vertical=False), T.RandomFlip(horizontal=False, vertical=True)]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs, recompute_boxes=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return SilkEvaluator(cfg.SILK.EVAL_SCORE_THR, log_csv=os.path.join(cfg.OUTPUT_DIR, "logs", "eval_log.csv"))

    def build_hooks(self):
        hs = super().build_hooks()
        hs.append(hooks.BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "silk/e2e_f1", "max", file_prefix="best"))
        return hs


def main() -> None:
    p = argparse.ArgumentParser(description="Train SOLOv2 trên data tằm")
    p.add_argument("--variant", choices=["r50", "mamba", "vmunet"], required=True)
    p.add_argument("--train-sets", nargs="+", default=None, help="vd v3:train dense_v2:train*8 (*N = lặp N lần); mặc định v3:train dense_v2:train*8")
    p.add_argument("--val-set", default=None, help="mặc định v3:valid")
    p.add_argument("--out-dir", default=None, help="thư mục chứa các run (mặc định runs/solov2_<variant>)")
    p.add_argument("--resume", default=None, help="thư mục run cần train tiếp")
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--img-size", type=int, default=None, help="cạnh ngắn khi chạy cả ảnh (valid/test), mặc định 640")
    p.add_argument("--crop-size", type=int, default=None, help="cửa sổ cắt khi train, mặc định 512")
    p.add_argument("--no-imagenet", action="store_true", help="r50: train từ đầu (để so công bằng với VSSM chưa có pretrained)")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--gpu", default="0")
    a = p.parse_args()

    c = SOLOv2Config(variant=a.variant)
    for k in ("max_iter", "batch_size", "lr", "img_size", "crop_size"):
        if getattr(a, k) is not None:
            setattr(c, k, getattr(a, k))
    if a.train_sets:
        c.train_sets = a.train_sets
    if a.val_set:
        c.val_set = a.val_set
    c.imagenet_r50 = not a.no_imagenet
    if a.resume:
        c.work_dir = a.resume
    elif a.smoke_test:
        c.work_dir = os.path.join("runs", f"solov2_{a.variant}", "smoke_test")
    elif a.out_dir:
        c.work_dir = os.path.join(a.out_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if a.max_iter and not a.smoke_test:                                    # giảm lr ở 2/3 và 7/8 số vòng (như 40000/52000 trên 60000)
        c.steps = [int(a.max_iter * 2 / 3), int(a.max_iter * 7 / 8)]
        c.eval_period = max(1000, a.max_iter // 8)
    if a.smoke_test:                                                       # --max-iter vẫn ghi đè được (vd thử học thuộc 64 ảnh)
        c.max_iter = a.max_iter or 20
        c.batch_size, c.warmup_iters = a.batch_size or 2, min(50, max(5, c.max_iter // 4))
        c.eval_period, c.steps = c.max_iter, [int(c.max_iter * 0.75), int(c.max_iter * 0.9)]

    torch.cuda.set_device(int(a.gpu))
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)
    train_names = [register(s, limit=64 if a.smoke_test else None) for s in c.train_sets]
    val_name = register(c.val_set, limit=8 if a.smoke_test else None)
    cfg = build_cfg(c, train_names, val_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(output=cfg.OUTPUT_DIR)
    if not a.resume:
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
            f.write(cfg.dump())
        with open(os.path.join(cfg.OUTPUT_DIR, "silk_config.yaml"), "w") as f:
            yaml.safe_dump(c.to_dict(), f, sort_keys=False, allow_unicode=True)

    trainer = Trainer(cfg)
    n_par = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    n_bb = sum(p.numel() for p in trainer.model.backbone.parameters()) / 1e6
    print(f"\n🐛 SOLOv2 [{a.variant}] | {n_par:.1f}M tham số (backbone {n_bb:.1f}M) | train {train_names} | val {val_name} | out {cfg.OUTPUT_DIR}\n")
    trainer.resume_or_load(resume=bool(a.resume))
    trainer.train()


if __name__ == "__main__":
    main()
