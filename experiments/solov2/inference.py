"""
inference.py — Nạp một run SOLOv2 và dự đoán trên MỘT ảnh RGB bất kỳ, chạy CẢ ảnh ở cỡ gần gốc (cạnh ngắn MIN_SIZE_TEST, mặc định 640; không cắt ô).

  pred = SolovPredictor("<run>/best.pth", gpu=1)
  instances = pred(rgb)                              # detectron2 Instances: pred_masks, pred_classes (0 = bệnh, 1 = Healthy), scores
  inst_map, cls_of = to_map(instances, 0.3, (H, W))  # bản đồ id [H, W] + {id: lớp}, mask điểm cao vẽ sau (đè lên)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import CfgNode
from detectron2.engine import DefaultPredictor

from experiments.solov2 import backbones  # noqa: F401
from experiments.solov2 import model as _model  # noqa: F401


class SolovPredictor:
    def __init__(self, checkpoint: str, gpu: int = 0):
        torch.cuda.set_device(gpu)
        run_dir = os.path.dirname(os.path.abspath(checkpoint))
        cfg = CfgNode(CfgNode.load_yaml_with_base(os.path.join(run_dir, "config.yaml")), new_allowed=True)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = checkpoint
        cfg.MODEL.DEVICE = f"cuda:{gpu}"
        cfg.freeze()
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray):
        return self.predictor(rgb[:, :, ::-1].copy())["instances"].to("cpu")           # DefaultPredictor nhận BGR


def to_map(instances, score_thr: float, size_hw: tuple[int, int]) -> tuple[np.ndarray, dict[int, int]]:
    """Mask điểm >= score_thr → bản đồ id kích thước size_hw (nearest), điểm tăng dần để mask điểm cao đè lên."""
    out = np.zeros(size_hw, np.int64)
    if len(instances) == 0 or instances.pred_masks.numel() == 0:
        return out, {}
    keep = (instances.scores >= score_thr).nonzero().flatten()
    if len(keep) == 0:
        return out, {}
    keep = keep[torch.argsort(instances.scores[keep])]
    masks = F.interpolate(instances.pred_masks[keep].float()[None], size=size_hw, mode="nearest")[0] > 0.5
    cls_of = {}
    for k, (m, c) in enumerate(zip(masks.numpy(), instances.pred_classes[keep].tolist()), 1):
        out[m] = k
        cls_of[k] = int(c)
    return out, {k: c for k, c in cls_of.items() if (out == k).any()}
