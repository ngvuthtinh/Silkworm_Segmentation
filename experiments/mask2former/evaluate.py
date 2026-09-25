"""
evaluate.py — Evaluation script for trained Mask2Former on Silkworm Mixed test set.
Computes mIoU, Dice, Precision, Recall, and Accuracy.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK2FORMER_ROOT = PROJECT_ROOT / "models" / "Mask2Former"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

from experiments.mask2former.dataset_adapter import register_silkworm_mixed_datasets, get_silkworm_mixed_dicts
from experiments.mask2former.config import setup_mask2former_cfg


def evaluate_test_set(checkpoint_path: str = "runs/mask2former/model_final.pth", output_dir: str = "runs/mask2former") -> dict:
    register_silkworm_mixed_datasets()
    cfg = setup_mask2former_cfg(output_dir=output_dir)
    cfg.MODEL.WEIGHTS = checkpoint_path

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)

    test_dicts = get_silkworm_mixed_dicts("test")
    print(f"Loaded {len(test_dicts)} test samples for evaluation.")

    ious = []
    dices = []
    precisions = []
    recalls = []

    transform = T.Resize((224, 224))

    with torch.no_grad():
        for record in tqdm(test_dicts, desc="Evaluating Mask2Former"):
            img = np.array(Image.open(record["file_name"]).convert("RGB"))
            gt_mask = np.array(Image.open(record["sem_seg_file_name"]))

            # Resize GT mask to (224, 224)
            gt_mask_resized = np.array(Image.fromarray(gt_mask).resize((224, 224), resample=Image.NEAREST))
            gt_binary = (gt_mask_resized > 0).astype(np.uint8)

            # Preprocess image
            aug_input = T.AugInput(img)
            transform(aug_input)
            img_tensor = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1)).to(cfg.MODEL.DEVICE)

            inputs = [{"image": img_tensor, "height": 224, "width": 224}]
            outputs = model(inputs)

            # Output semantic segmentation map
            sem_seg = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()
            pred_binary = (sem_seg == 1).astype(np.uint8)

            # Compute intersection & union
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()

            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2.0 * intersection + 1e-6) / (pred_binary.sum() + gt_binary.sum() + 1e-6)
            precision = (intersection + 1e-6) / (pred_binary.sum() + 1e-6)
            recall = (intersection + 1e-6) / (gt_binary.sum() + 1e-6)

            ious.append(iou)
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)

    results = {
        "model": "Mask2Former",
        "checkpoint": checkpoint_path,
        "num_test_samples": len(test_dicts),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
    }

    out_file = Path(output_dir) / "test_evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 50)
    print("📊 Mask2Former Evaluation Results on Silkworm Test Set:")
    print(f"  • Mean IoU:       {results['mean_iou'] * 100:.2f}%")
    print(f"  • Mean Dice:      {results['mean_dice'] * 100:.2f}%")
    print(f"  • Mean Precision: {results['mean_precision'] * 100:.2f}%")
    print(f"  • Mean Recall:    {results['mean_recall'] * 100:.2f}%")
    print("=" * 50)

    return results


if __name__ == "__main__":
    ckpt = "runs/mask2former/model_final.pth"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    evaluate_test_set(checkpoint_path=ckpt)
