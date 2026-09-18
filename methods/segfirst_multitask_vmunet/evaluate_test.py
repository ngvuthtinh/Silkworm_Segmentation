"""
evaluate_test.py — Comprehensive Test Evaluation & Visualization for SegFirst Multi-Task VM-UNet.

Evaluates the model on:
  1. Single-silkworm test subset (100 images from SAM3)
  2. Multi-silkworm test subset (69 images from legacy SilkyNet)
  3. Combined overall test set (169 images)

Computes Dice Score, IoU, Precision, Recall, and Classification Accuracy.
Generates 4-panel visual comparisons with metrics overlays.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

# Resolve paths
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from config import SegFirstConfig
from model import SegFirstVMUNet

CLASS_NAMES = {0: "Healthy", 1: "Diseased"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SegFirst Multi-Task VM-UNet on Test Set")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="runs/segfirst_vmunet/2026-09-17_run4/checkpoints/phase2_best.pth",
        help="Path to model checkpoint (.pth)",
    )
    p.add_argument(
        "--test-dir",
        type=str,
        default="data/silkworm_mixed_dataset/test",
        help="Path to test set directory containing images/, masks/, labels/",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="runs/segfirst_vmunet/2026-09-17_run4/test_evaluation",
        help="Directory to save evaluation results and visualizations",
    )
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--gpu", type=str, default="auto")
    p.add_argument("--save-vis-count", type=int, default=15, help="Number of visualizations per subset")
    return p.parse_args()


class TestEvaluationDataset(Dataset):
    def __init__(self, test_dir: str, img_size: int = 128):
        self.img_size = img_size
        self.root = Path(test_dir)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.lbl_dir = self.root / "labels"

        self.samples = []
        for img_p in sorted(self.img_dir.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = img_p.stem
            mask_p = self.mask_dir / f"{stem}.png"
            lbl_p = self.lbl_dir / f"{stem}.txt"

            is_single = "healthy" in stem.lower() or "grasserie" in stem.lower()
            
            # Extract ground truth class if available from filename or label file
            gt_class = None
            if "healthy" in stem.lower():
                gt_class = 0
            elif "grasserie" in stem.lower():
                gt_class = 1
            elif lbl_p.exists():
                lines = [l.strip() for l in lbl_p.read_text().splitlines() if l.strip()]
                if lines:
                    cls_id = int(lines[0].split()[0])
                    # In SAM3 YOLO: 0 is Grasserie (diseased), 1 is Healthy
                    gt_class = 1 if cls_id == 0 else 0

            self.samples.append({
                "stem": stem,
                "img_path": img_p,
                "mask_path": mask_p if mask_p.exists() else None,
                "lbl_path": lbl_p if lbl_p.exists() else None,
                "is_single": is_single,
                "gt_class": gt_class,
            })

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_pil = Image.open(item["img_path"]).convert("RGB")
        orig_w, orig_h = img_pil.size

        img_resized = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = self.norm(torch.from_numpy(img_arr.transpose(2, 0, 1)))

        if item["mask_path"] is not None:
            mask_pil = Image.open(item["mask_path"]).convert("L")
            mask_full = (np.array(mask_pil, dtype=np.float32) > 0).astype(np.float32)
            mask_resized = (np.array(mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)) > 0).astype(np.float32)
        else:
            mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)
            mask_resized = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        return {
            "img_tensor": img_tensor,
            "mask_tensor": torch.from_numpy(mask_resized).unsqueeze(0),
            "orig_img": np.array(img_pil),
            "orig_mask": mask_full,
            "stem": item["stem"],
            "is_single": item["is_single"],
            "gt_class": item["gt_class"] if item["gt_class"] is not None else -1,
        }


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict[str, float]:
    """Computes binary segmentation metrics: Dice, IoU, Precision, Recall."""
    intersection = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())

    iou = intersection / union if union > 0 else (1.0 if gt_sum == 0 and pred_sum == 0 else 0.0)
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else (1.0 if gt_sum == 0 and pred_sum == 0 else 0.0)
    precision = intersection / pred_sum if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = intersection / gt_sum if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def save_visualization(
    save_path: Path,
    orig_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    metrics: dict[str, float],
    pred_cls_name: str,
    pred_cls_conf: float,
    gt_cls_name: str,
    title_suffix: str = "",
):
    """Saves a clean 4-panel visual comparison plot."""
    h, w = orig_img.shape[:2]
    pred_mask_resized = np.array(
        Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    ) > 0

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # 1. Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original Image\nGT: {gt_cls_name}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Predicted Mask
    axes[2].imshow(pred_mask_resized, cmap="gray")
    axes[2].set_title(
        f"Predicted Mask\nDice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f}",
        fontsize=12,
        fontweight="bold",
        color="green" if metrics['dice'] >= 0.7 else "darkorange",
    )
    axes[2].axis("off")

    # 4. Color Overlay
    overlay = orig_img.copy().astype(np.float32) / 255.0
    # Purple/Cyan overlay for silkworm segmentation
    overlay[pred_mask_resized, 0] = overlay[pred_mask_resized, 0] * 0.4 + 0.6
    overlay[pred_mask_resized, 1] = overlay[pred_mask_resized, 1] * 0.4 + 0.1
    overlay[pred_mask_resized, 2] = overlay[pred_mask_resized, 2] * 0.4 + 0.8
    overlay = np.clip(overlay, 0.0, 1.0)

    axes[3].imshow(overlay)
    cls_status = "CORRECT" if (gt_cls_name == pred_cls_name or gt_cls_name == "Unknown") else "WRONG"
    axes[3].set_title(
        f"Segment Overlay\nPred: {pred_cls_name} ({pred_cls_conf*100:.1f}%) [{cls_status}]",
        fontsize=12,
        fontweight="bold",
    )
    axes[3].axis("off")

    plt.suptitle(f"{title_suffix}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    vis_single_dir = out_dir / "visualizations_single_silkworm"
    vis_multi_dir = out_dir / "visualizations_multi_silkworm"
    vis_single_dir.mkdir(parents=True, exist_ok=True)
    vis_multi_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        if args.gpu == "auto":
            best_gpu = 0
            max_free = -1
            for i in range(torch.cuda.device_count()):
                f, t = torch.cuda.mem_get_info(i)
                if f > max_free:
                    max_free = f
                    best_gpu = i
            device = torch.device(f"cuda:{best_gpu}")
        else:
            device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("================================================================")
    print(" SegFirst Multi-Task VM-UNet Comprehensive Test Evaluation")
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Test Dir   : {args.test_dir}")
    print(f" Output Dir : {args.output_dir}")
    print(f" Device     : {device}")
    print("================================================================")

    # Load Model
    model = SegFirstVMUNet(input_channels=3, num_seg_classes=1, num_cls_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load Dataset
    dataset = TestEvaluationDataset(args.test_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"[INFO] Loaded {len(dataset)} test samples.")

    # Storage for group metrics
    results = {
        "all": {"dice": [], "iou": [], "precision": [], "recall": []},
        "single": {"dice": [], "iou": [], "precision": [], "recall": []},
        "multi": {"dice": [], "iou": [], "precision": [], "recall": []},
    }

    cls_eval = {
        "total_labeled": 0,
        "correct": 0,
        "y_true": [],
        "y_pred": [],
    }

    saved_single_count = 0
    saved_multi_count = 0

    with torch.no_grad():
        for item in tqdm(loader, desc="Evaluating Test Set", ncols=100):
            imgs = item["img_tensor"].to(device)
            mask_logits, class_logits = model(imgs, phase=2)
            pred_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
            pred_bin = (pred_prob >= 0.5).astype(np.uint8)

            orig_img = item["orig_img"][0].numpy()
            orig_mask = item["orig_mask"][0].numpy()
            stem = item["stem"][0]
            is_single = item["is_single"][0].item()
            gt_class = item["gt_class"][0].item()

            # Resize pred_bin to match ground truth dimensions for evaluation
            h, w = orig_mask.shape[:2]
            pred_bin_full = np.array(
                Image.fromarray((pred_bin * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
            ) > 0

            # Compute segmentation metrics
            m = compute_metrics(pred_bin_full, orig_mask > 0)
            
            # Store in 'all'
            for k in ["dice", "iou", "precision", "recall"]:
                results["all"][k].append(m[k])

            # Store in subset
            subset_key = "single" if is_single else "multi"
            for k in ["dice", "iou", "precision", "recall"]:
                results[subset_key][k].append(m[k])

            # Classification prediction
            pred_cls_name = "N/A"
            pred_cls_conf = 0.0
            if class_logits is not None:
                probs = torch.softmax(class_logits, dim=1).squeeze().cpu().numpy()
                pred_c = int(np.argmax(probs))
                pred_cls_conf = float(probs[pred_c])
                pred_cls_name = CLASS_NAMES[pred_c]

                if gt_class in [0, 1]:
                    cls_eval["total_labeled"] += 1
                    if pred_c == gt_class:
                        cls_eval["correct"] += 1
                    cls_eval["y_true"].append(gt_class)
                    cls_eval["y_pred"].append(pred_c)

            gt_cls_name = CLASS_NAMES[gt_class] if gt_class in [0, 1] else "Unknown"

            # Save sample visualizations
            if is_single and saved_single_count < args.save_vis_count:
                save_visualization(
                    vis_single_dir / f"single_{saved_single_count:02d}_{stem}.png",
                    orig_img,
                    orig_mask,
                    pred_bin,
                    m,
                    pred_cls_name,
                    pred_cls_conf,
                    gt_cls_name,
                    title_suffix=f"Single Silkworm Test Sample [{stem}]",
                )
                saved_single_count += 1
            elif not is_single and saved_multi_count < args.save_vis_count:
                save_visualization(
                    vis_multi_dir / f"multi_{saved_multi_count:02d}_{stem}.png",
                    orig_img,
                    orig_mask,
                    pred_bin,
                    m,
                    pred_cls_name,
                    pred_cls_conf,
                    gt_cls_name,
                    title_suffix=f"Multi Silkworm Test Sample [{stem}]",
                )
                saved_multi_count += 1

    # Summarize Metrics
    summary = {}
    for grp in ["single", "multi", "all"]:
        summary[grp] = {
            "count": len(results[grp]["dice"]),
            "dice_mean": float(np.mean(results[grp]["dice"])),
            "dice_std": float(np.std(results[grp]["dice"])),
            "iou_mean": float(np.mean(results[grp]["iou"])),
            "iou_std": float(np.std(results[grp]["iou"])),
            "precision_mean": float(np.mean(results[grp]["precision"])),
            "recall_mean": float(np.mean(results[grp]["recall"])),
        }

    cls_acc = (
        cls_eval["correct"] / max(1, cls_eval["total_labeled"])
        if cls_eval["total_labeled"] > 0
        else 0.0
    )
    summary["classification"] = {
        "total_labeled_samples": cls_eval["total_labeled"],
        "accuracy": float(cls_acc),
        "correct": cls_eval["correct"],
    }

    # Save summary JSON
    with open(out_dir / "test_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # Print Formatted Report
    print("\n" + "=" * 70)
    print("         EVALUATION RESULTS SUMMARY (TEST DATASET)")
    print("=" * 70)
    print(f" Model Checkpoint: {args.checkpoint}")
    print(f" Total Samples Tested: {summary['all']['count']}")
    print("-" * 70)
    print(f"{'Subset':<20} | {'Count':<6} | {'mDice (%)':<12} | {'mIoU (%)':<12} | {'Precision (%)':<14} | {'Recall (%)':<10}")
    print("-" * 70)
    for grp, name in [("single", "Single Silkworm"), ("multi", "Multi Silkworm"), ("all", "Overall Test Set")]:
        d = summary[grp]
        print(
            f"{name:<20} | {d['count']:<6} | {d['dice_mean']*100:<12.2f} | {d['iou_mean']*100:<12.2f} | {d['precision_mean']*100:<14.2f} | {d['recall_mean']*100:<10.2f}"
        )
    print("-" * 70)
    if cls_eval["total_labeled"] > 0:
        print(f" Disease Classification Accuracy: {cls_acc*100:.2f}% ({cls_eval['correct']}/{cls_eval['total_labeled']} correct)")
    print(f" Visualizations saved to:\n  - Single: {vis_single_dir}\n  - Multi : {vis_multi_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
