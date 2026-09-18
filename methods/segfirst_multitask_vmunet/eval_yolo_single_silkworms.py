#!/usr/bin/env python3
"""
eval_yolo_single_silkworms.py — Benchmark VM-UNet model on YOLO single silkworm test set.
Evaluates SegFirst Multi-Task VM-UNet on data/Silkworm Diseases.v1i.yolo26/test (498 samples).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from config import SegFirstConfig
from model import SegFirstVMUNet

CLASS_NAMES = {0: "Healthy", 1: "Grasserie"}


def load_model(ckpt_path: str, device: torch.device, img_size: int = 128) -> torch.nn.Module:
    model = SegFirstVMUNet(input_channels=3, num_seg_classes=1, num_cls_classes=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def parse_yolo_annotation(txt_path: str, orig_w: int, orig_h: int) -> tuple[int, list[list[int]], np.ndarray]:
    bbox_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    boxes = []
    gt_class = None

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            c_raw = int(parts[0])
            c = 1 if c_raw == 0 else 0
            if gt_class is None:
                gt_class = c
            xc, yc, w, h = map(float, parts[1:5])
            x1 = max(0, int(round((xc - w / 2.0) * orig_w)))
            y1 = max(0, int(round((yc - h / 2.0) * orig_h)))
            x2 = min(orig_w, int(round((xc + w / 2.0) * orig_w)))
            y2 = min(orig_h, int(round((yc + h / 2.0) * orig_h)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                bbox_mask[y1:y2, x1:x2] = 1

    return gt_class, boxes, bbox_mask


def compute_bbox_iou(boxA: list[int], boxB: list[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    return float(interArea / denom) if denom > 0 else 0.0


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
    return [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices)), int(np.max(y_indices))]


def evaluate_model(
    model: torch.nn.Module,
    test_items: list[tuple[str, str, int]],
    device: torch.device,
    img_size: int = 128,
) -> tuple[dict, list[dict]]:
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    records = []
    latencies = []
    containments = []
    bbox_ious = []
    area_ratios = []

    y_true_all = []
    y_pred_all = []

    for img_path, txt_path, fallback_cls in test_items:
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        gt_cls, gt_boxes, gt_bbox_mask = parse_yolo_annotation(txt_path, orig_w, orig_h)
        if gt_cls is None:
            gt_cls = fallback_cls

        img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = norm(torch.from_numpy(img_arr.transpose(2, 0, 1))).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            mask_logits, class_logits = model(tensor, phase=2)
            prob_tensor = torch.sigmoid(mask_logits)
            cls_probs = torch.softmax(class_logits, dim=1).squeeze().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        pred_cls = int(np.argmax(cls_probs))
        pred_conf = float(cls_probs[pred_cls])

        y_true_all.append(gt_cls)
        y_pred_all.append(pred_cls)

        prob_arr = prob_tensor.squeeze().cpu().numpy()
        prob_full = np.array(
            Image.fromarray((prob_arr * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        ) / 255.0
        mask_bin = (prob_full >= 0.5).astype(np.uint8)

        mask_pixels = int(np.sum(mask_bin))
        total_pixels = orig_w * orig_h
        area_ratio = mask_pixels / total_pixels
        area_ratios.append(area_ratio)

        # Containment inside GT bounding boxes
        if mask_pixels > 0 and len(gt_boxes) > 0:
            inside_pixels = int(np.sum(mask_bin * gt_bbox_mask))
            containment = inside_pixels / mask_pixels
        else:
            containment = 1.0 if mask_pixels == 0 and len(gt_boxes) == 0 else 0.0
        containments.append(containment)

        # BBox IoU
        pred_box = mask_to_bbox(mask_bin)
        if pred_box is not None and len(gt_boxes) > 0:
            best_iou = max(compute_bbox_iou(pred_box, gb) for gb in gt_boxes)
        else:
            best_iou = 0.0
        bbox_ious.append(best_iou)

        records.append({
            "image_path": img_path,
            "filename": os.path.basename(img_path),
            "gt_class": gt_cls,
            "pred_class": pred_cls,
            "pred_conf": pred_conf,
            "correct": int(pred_cls == gt_cls),
            "containment": float(containment),
            "bbox_iou": float(best_iou),
            "area_ratio": float(area_ratio),
            "gt_boxes": gt_boxes,
            "pred_box": pred_box,
        })

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    # Classification metrics
    accuracy = float(np.mean(y_true == y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # Grasserie predicted as Grasserie
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # Healthy predicted as Grasserie
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # Grasserie predicted as Healthy
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # Healthy predicted as Healthy

    prec_g = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_g = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_g = float(2 * prec_g * rec_g / (prec_g + rec_g)) if (prec_g + rec_g) > 0 else 0.0

    prec_h = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_h = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_h = float(2 * prec_h * rec_h / (prec_h + rec_h)) if (prec_h + rec_h) > 0 else 0.0

    macro_f1 = float((f1_g + f1_h) / 2.0)

    summary = {
        "num_samples": len(test_items),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "grasserie": {"precision": prec_g, "recall": rec_g, "f1": f1_g, "support": int(np.sum(y_true == 1))},
        "healthy": {"precision": prec_h, "recall": rec_h, "f1": f1_h, "support": int(np.sum(y_true == 0))},
        "mean_containment": float(np.mean(containments)),
        "mean_bbox_iou": float(np.mean(bbox_ious)),
        "mean_area_ratio": float(np.mean(area_ratios)),
        "mean_latency_ms": float(np.mean(latencies)),
    }

    return summary, records


def main():
    parser = argparse.ArgumentParser(description="Evaluate VM-UNet model on YOLO single silkworms")
    parser.add_argument("--yolo-dir", default="data/Silkworm Diseases.v1i.yolo26", help="Path to YOLO dataset")
    parser.add_argument("--ckpt", default="runs/segfirst_vmunet/2026-09-17_run4/checkpoints/phase2_best.pth")
    parser.add_argument("--out-json", default="runs/segfirst_vmunet/yolo_eval_vmunet.json")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    gpu_idx = int(args.gpu) if args.gpu.isdigit() else 0
    if torch.cuda.is_available():
        gpu_idx = min(gpu_idx, torch.cuda.device_count() - 1)
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    img_dir = os.path.join(args.yolo_dir, "test/images")
    lbl_dir = os.path.join(args.yolo_dir, "test/labels")

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"\n==================================================================")
    print(f" YOLO Single Silkworms Benchmark (SegFirst Multi-Task VM-UNet)")
    print(f" Test images: {len(image_files)} in {img_dir}")
    print(f" Device: {device}")
    print(f"==================================================================\n")

    test_items = []
    for img_p in image_files:
        base = os.path.splitext(os.path.basename(img_p))[0]
        txt_p = os.path.join(lbl_dir, base + ".txt")
        fallback_cls = 0 if base.lower().startswith("healthy") else 1
        test_items.append((img_p, txt_p, fallback_cls))

    results = {}
    if os.path.exists(args.ckpt):
        print(f"Evaluating SegFirst Multi-Task VM-UNet from {args.ckpt}...")
        model = load_model(args.ckpt, device, img_size=args.img_size)
        summary, recs = evaluate_model(model, test_items, device, img_size=args.img_size)
        results["segfirst_vmunet"] = {
            "checkpoint": args.ckpt,
            "summary": summary,
            "records": recs,
        }
        print(f"  Accuracy:         {summary['accuracy']*100:.2f}%")
        print(f"  Macro F1:         {summary['macro_f1']*100:.2f}%")
        print(f"  Healthy F1:       {summary['healthy']['f1']*100:.2f}% (P={summary['healthy']['precision']*100:.1f}%, R={summary['healthy']['recall']*100:.1f}%)")
        print(f"  Grasserie F1:     {summary['grasserie']['f1']*100:.2f}% (P={summary['grasserie']['precision']*100:.1f}%, R={summary['grasserie']['recall']*100:.1f}%)")
        print(f"  Confusion Matrix: TP={summary['confusion_matrix']['tp']}, FP={summary['confusion_matrix']['fp']}, FN={summary['confusion_matrix']['fn']}, TN={summary['confusion_matrix']['tn']}")
        print(f"  BBox Containment: {summary['mean_containment']*100:.2f}%")
        print(f"  BBox IoU:         {summary['mean_bbox_iou']*100:.2f}%")
        print(f"  Latency:          {summary['mean_latency_ms']:.2f} ms/img")

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved VM-UNet evaluation results to {out_p}")


if __name__ == "__main__":
    main()
