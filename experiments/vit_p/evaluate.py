"""
evaluate.py — Evaluation script for trained ViT-P on Silkworm Mixed test set.
Computes classification accuracy, precision, recall, and per-class metrics on test set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vit_p.config import ViTPConfig
from experiments.vit_p.dataset_adapter import SilkwormViTPDataset
from experiments.vit_p.model import ViTPSegClassifier


def evaluate(checkpoint_path: str, gpu_id: str = "0"):
    cfg = ViTPConfig()
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 70)
    print(f"📊 Evaluating ViT-P on Silkworm Mixed Test Set")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Device    : {device_str}")
    print("=" * 70)

    # 1. Dataset & DataLoader
    test_dataset = SilkwormViTPDataset(
        split_dir=os.path.join(cfg.data_dir, "test"),
        input_size=cfg.input_size,
        num_points=cfg.num_points,
        is_train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 2. Build Model & Load Checkpoint
    model = ViTPSegClassifier(
        arch=cfg.arch,
        num_classes=cfg.num_classes,
        num_points=cfg.num_points,
        img_size=cfg.input_size,
        patch_size=cfg.patch_size,
        pretrained=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_targets = []

    pbar = tqdm(test_loader, desc="Testing 169 images", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images, points)

            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(labels.view(-1).cpu().numpy())

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    # Metrics for Class 1 (Silkworm) and Overall Accuracy
    tp = int(np.sum((preds_np == 1) & (targets_np == 1)))
    fp = int(np.sum((preds_np == 1) & (targets_np == 0)))
    fn = int(np.sum((preds_np == 0) & (targets_np == 1)))
    tn = int(np.sum((preds_np == 0) & (targets_np == 0)))

    acc = np.mean(preds_np == targets_np) * 100.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    iou = (tp / (tp + fp + fn)) * 100.0 if (tp + fp + fn) > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"🎯 ViT-P Test Evaluation Results (169 Test Images, {len(preds_np)} points):")
    print(f"   Accuracy    : {acc:.2f}%")
    print(f"   IoU-Silkworm: {iou:.2f}%")
    print(f"   Precision   : {prec:.2f}%")
    print(f"   Recall      : {rec:.2f}%")
    print(f"   F1-Score    : {f1:.2f}%")
    print("-" * 60)
    print(f"   Confusion Matrix:")
    print(f"     True Positive (TP - Tằm đúng): {tp:5d}  | False Positive (FP - Nền đoán nhầm): {fp:5d}")
    print(f"     False Negative (FN - Bỏ sót):  {fn:5d}  | True Negative (TN - Nền đúng):       {tn:5d}")
    print("=" * 60)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/vit_p/checkpoints/best_model.pth"
    evaluate(ckpt)
