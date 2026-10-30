#!/usr/bin/env python3
"""
experiments/solov2/evaluate.py — Chấm một checkpoint SOLOv2 theo từng con (cùng cách chấm với VM-UNet; xem evaluator.py).

  python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split v3:valid --score-thrs 0.2 0.3 0.4 0.5   # chọn ngưỡng trên valid
  python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split v3:test --score-thrs 0.3                 # test một lần
  python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split v3:test_real
  python -m experiments.solov2.evaluate --checkpoint <run>/best.pth --split dense_v2:test                         # ảnh dày (L2)
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model

from experiments.solov2 import backbones  # noqa: F401
from experiments.solov2 import model as _model  # noqa: F401
from experiments.solov2.data import register


def load_cfg(checkpoint: str) -> CfgNode:
    run_dir = os.path.dirname(os.path.abspath(checkpoint))
    cfg = CfgNode(CfgNode.load_yaml_with_base(os.path.join(run_dir, "config.yaml")), new_allowed=True)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="v3:test", help="<tên data>:<split>, vd v3:test, v3:test_real, dense_v2:test")
    p.add_argument("--score-thrs", type=float, nargs="+", default=None, help="mặc định = ngưỡng lưu trong config của run")
    p.add_argument("--limit", type=int, default=None, help="chỉ chấm N ảnh đầu (chạy thử)")
    p.add_argument("--gpu", default="0")
    a = p.parse_args()

    torch.cuda.set_device(int(a.gpu))
    cfg = load_cfg(a.checkpoint)
    name = register(a.split, limit=a.limit)
    cfg.defrost()
    cfg.DATASETS.TEST = (name,)
    cfg.freeze()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(a.checkpoint)
    model.eval()
    from experiments.solov2.evaluator import SilkEvaluator
    thrs = a.score_thrs or [float(cfg.SILK.EVAL_SCORE_THR)]
    res = inference_on_dataset(model, build_detection_test_loader(cfg, name), SilkEvaluator(thrs))["silk"]
    print(f"\n[{a.split}] {name} | backbone {cfg.SILK.VARIANT} | score_thr {thrs}")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "by_score_thr"}, indent=2))
    for t, r in res.get("by_score_thr", {}).items():
        print(f"  score_thr {t}: e2e_f1 {r['e2e_f1']:.4f} | inst_f1 {r['inst_f1']:.4f} | grasserie_recall {r['grasserie_recall']:.4f} | vùng {r['n_regions']}")


if __name__ == "__main__":
    main()
