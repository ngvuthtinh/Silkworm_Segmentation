#!/usr/bin/env python3
"""
eval_yolo_single_silkworms.py — Benchmark Swin-Unet models on YOLO single silkworm test set.
Evaluates both:
  1. SegFirst Multi-Task Swin-Unet
  2. Joint Multi-Task Swin-Unet
on data/Silkworm Diseases.v1i.yolo26/test (498 samples: 256 Healthy, 242 Grasserie).
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
import torchvision.transforms.functional as TF

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from config import SegFirstSwinConfig
from model import SegFirstSwinUnet

CLASS_NAMES = {0: "Healthy", 1: "Grasserie"}


def load_model(ckpt_path: str, device: torch.device, img_size: int = 224) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    cleaned_state = {}
    for k, v in state.items():
        clean_k = k.replace("module.", "")
        cleaned_state[clean_k] = v

    num_seg_classes = 1
    if "swin_unet.output.weight" in cleaned_state:
        num_seg_classes = cleaned_state["swin_unet.output.weight"].shape[0]

    cfg = SegFirstSwinConfig(input_size=img_size, num_seg_classes=num_seg_classes)
    model = SegFirstSwinUnet(
        config=cfg,
        input_size=img_size,
        num_seg_classes=num_seg_classes,
        num_cls_classes=2,
    ).to(device)

    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    return model, num_seg_classes


def parse_yolo_annotation(txt_path: str, orig_w: int, orig_h: int) -> tuple[int, list[list[int]], np.ndarray]:
    """
    Returns:
      gt_class (0 for Healthy, 1 for Grasserie),
      list of bboxes [[x1, y1, x2, y2], ...],
      binary bbox mask of shape (orig_h, orig_w)
    """
    bbox_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    boxes = []
    gt_class = None

    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            c_raw = int(parts[0])
            # YOLO classes: 0 = Grasserie, 1 = Healthy
            # Our model classes: 0 = Healthy, 1 = Grasserie
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
    num_seg_classes: int,
    test_items: list[tuple[str, str, int]],
    device: torch.device,
    img_size: int = 224,
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

        img_resized = TF.resize(img_pil, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = norm(TF.to_tensor(img_resized)).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            mask_logits, class_logits = model(tensor, phase=2)
            if num_seg_classes == 2:
                prob_tensor = torch.softmax(mask_logits, dim=1)[:, 1, :, :]
            else:
                prob_tensor = torch.sigmoid(mask_logits)
            cls_probs = torch.softmax(class_logits, dim=1).squeeze().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        pred_cls = int(np.argmax(cls_probs))
        pred_conf = float(cls_probs[pred_cls])

        y_true_all.append(gt_cls)
        y_pred_all.append(pred_cls)

        # Upscale probability to original image size
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

    # Confusion matrix:
    # 0 = Healthy, 1 = Grasserie
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
    parser = argparse.ArgumentParser(description="Evaluate Swin-Unet models on YOLO single silkworms")
    parser.add_argument("--yolo-dir", default="data/Silkworm Diseases.v1i.yolo26", help="Path to YOLO dataset")
    parser.add_argument("--segfirst-ckpt", default="methods/segfirst_multitask_swinunet/results/run_20260917_025501/checkpoints/phase2_best.pth")
    parser.add_argument("--joint-ckpt", default="Swin-Unet/multitask_silkworm_out/best_model.pth")
    parser.add_argument("--out-json", default="methods/segfirst_multitask_swinunet/results/yolo_eval_swin.json")
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
    print(f" YOLO Single Silkworms Benchmark (Swin-Unet Models)")
    print(f" Test images: {len(image_files)} in {img_dir}")
    print(f" Device: {device}")
    print(f"==================================================================\n")

    test_items = []
    for img_p in image_files:
        base = os.path.splitext(os.path.basename(img_p))[0]
        txt_p = os.path.join(lbl_dir, base + ".txt")
        # Fallback class from filename prefix:
        # Healthy -> 0, Grasserie -> 1
        fallback_cls = 0 if base.lower().startswith("healthy") else 1
        test_items.append((img_p, txt_p, fallback_cls))

    results = {}

    # 1. Evaluate SegFirst Swin-Unet
    if os.path.exists(args.segfirst_ckpt):
        print(f"\nEvaluating SegFirst Multi-Task Swin-Unet from {args.segfirst_ckpt}...")
        model_sf, num_seg = load_model(args.segfirst_ckpt, device, img_size=224)
        summary_sf, recs_sf = evaluate_model(model_sf, num_seg, test_items, device, img_size=224)
        results["segfirst_swinunet"] = {
            "checkpoint": args.segfirst_ckpt,
            "summary": summary_sf,
            "records": recs_sf,
        }
        print(f"  Accuracy:         {summary_sf['accuracy']*100:.2f}%")
        print(f"  Macro F1:         {summary_sf['macro_f1']*100:.2f}%")
        print(f"  Healthy F1:       {summary_sf['healthy']['f1']*100:.2f}% (P={summary_sf['healthy']['precision']*100:.1f}%, R={summary_sf['healthy']['recall']*100:.1f}%)")
        print(f"  Grasserie F1:     {summary_sf['grasserie']['f1']*100:.2f}% (P={summary_sf['grasserie']['precision']*100:.1f}%, R={summary_sf['grasserie']['recall']*100:.1f}%)")
        print(f"  Confusion Matrix: TP={summary_sf['confusion_matrix']['tp']}, FP={summary_sf['confusion_matrix']['fp']}, FN={summary_sf['confusion_matrix']['fn']}, TN={summary_sf['confusion_matrix']['tn']}")
        print(f"  BBox Containment: {summary_sf['mean_containment']*100:.2f}%")
        print(f"  BBox IoU:         {summary_sf['mean_bbox_iou']*100:.2f}%")
        print(f"  Latency:          {summary_sf['mean_latency_ms']:.2f} ms/img")
        del model_sf
        torch.cuda.empty_cache()

    # 2. Evaluate Joint Swin-Unet
    if os.path.exists(args.joint_ckpt):
        print(f"\nEvaluating Joint Multi-Task Swin-Unet from {args.joint_ckpt}...")
        model_jt, num_seg = load_model(args.joint_ckpt, device, img_size=224)
        summary_jt, recs_jt = evaluate_model(model_jt, num_seg, test_items, device, img_size=224)
        results["joint_swinunet"] = {
            "checkpoint": args.joint_ckpt,
            "summary": summary_jt,
            "records": recs_jt,
        }
        print(f"  Accuracy:         {summary_jt['accuracy']*100:.2f}%")
        print(f"  Macro F1:         {summary_jt['macro_f1']*100:.2f}%")
        print(f"  Healthy F1:       {summary_jt['healthy']['f1']*100:.2f}% (P={summary_jt['healthy']['precision']*100:.1f}%, R={summary_jt['healthy']['recall']*100:.1f}%)")
        print(f"  Grasserie F1:     {summary_jt['grasserie']['f1']*100:.2f}% (P={summary_jt['grasserie']['precision']*100:.1f}%, R={summary_jt['grasserie']['recall']*100:.1f}%)")
        print(f"  Confusion Matrix: TP={summary_jt['confusion_matrix']['tp']}, FP={summary_jt['confusion_matrix']['fp']}, FN={summary_jt['confusion_matrix']['fn']}, TN={summary_jt['confusion_matrix']['tn']}")
        print(f"  BBox Containment: {summary_jt['mean_containment']*100:.2f}%")
        print(f"  BBox IoU:         {summary_jt['mean_bbox_iou']*100:.2f}%")
        print(f"  Latency:          {summary_jt['mean_latency_ms']:.2f} ms/img")
        del model_jt
        torch.cuda.empty_cache()

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Swin-Unet evaluation results to {out_p}")


if __name__ == "__main__":
    main()
