#!/usr/bin/env python3
"""
generate_yolo_comparisons.py — Comparative Analysis & Visualizations for 3 Models on YOLO Single Silkworms.

Models:
  1. SegFirst Multi-Task Swin-Unet
  2. SegFirst Multi-Task VM-UNet
  3. Joint Multi-Task Swin-Unet

Produces:
  - Formatted comparison table
  - Side-by-side confusion matrix plot
  - Multi-panel visual comparisons for Healthy and Grasserie single silkworms
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

CLASS_NAMES = {0: "Healthy", 1: "Grasserie"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}  # Green, Red


def parse_yolo_boxes(txt_path: str, orig_w: int, orig_h: int) -> list[list[int]]:
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            xc, yc, w, h = map(float, parts[1:5])
            x1 = max(0, int(round((xc - w / 2.0) * orig_w)))
            y1 = max(0, int(round((yc - h / 2.0) * orig_h)))
            x2 = min(orig_w, int(round((xc + w / 2.0) * orig_w)))
            y2 = min(orig_h, int(round((yc + h / 2.0) * orig_h)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
    return boxes


def create_purple_overlay(img_np: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = img_np.copy().astype(np.float32)
    purple_color = np.array([142, 68, 173], dtype=np.float32)  # Amethyst Purple #8E44AD
    mask_bool = mask_bin.astype(bool)
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * purple_color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_confusion_matrices(results: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), dpi=150)
    if len(results) == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        cm = data["summary"]["confusion_matrix"]
        matrix = np.array([
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]]
        ])
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{model_name}\nAcc: {data['summary']['accuracy']*100:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Healthy", "Pred Grasserie"], fontsize=9)
        ax.set_yticklabels(["True Healthy", "True Grasserie"], fontsize=9)

        # Annotations
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swin-json", default="methods/segfirst_multitask_swinunet/results/yolo_eval_swin.json")
    parser.add_argument("--vmunet-json", default="runs/segfirst_vmunet/yolo_eval_vmunet.json")
    parser.add_argument("--out-dir", default="methods/segfirst_multitask_swinunet/results/yolo_visualizations")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if os.path.exists(args.swin_json):
        with open(args.swin_json) as f:
            data = json.load(f)
            if "segfirst_swinunet" in data:
                results["SegFirst Swin-Unet"] = data["segfirst_swinunet"]
            if "joint_swinunet" in data:
                results["Joint Swin-Unet"] = data["joint_swinunet"]

    if os.path.exists(args.vmunet_json):
        with open(args.vmunet_json) as f:
            data = json.load(f)
            if "segfirst_vmunet" in data:
                results["SegFirst VM-UNet"] = data["segfirst_vmunet"]

    if not results:
        print("No evaluation results found yet!")
        return

    # Print Summary Table
    print("\n" + "=" * 95)
    print(" 🏆 YOLO SINGLE SILKWORMS TEST SET BENCHMARK (498 Samples)")
    print("=" * 95)
    header = f"{'Model':<24} | {'Acc (%)':<8} | {'Macro F1':<9} | {'Healthy F1':<11} | {'Gras. F1':<10} | {'Contain%':<9} | {'BBox IoU':<9} | {'Latency':<8}"
    print(header)
    print("-" * 95)

    for name, data in results.items():
        s = data["summary"]
        row = (
            f"{name:<24} | "
            f"{s['accuracy']*100:>6.2f}% | "
            f"{s['macro_f1']*100:>7.2f}% | "
            f"{s['healthy']['f1']*100:>9.2f}% | "
            f"{s['grasserie']['f1']*100:>8.2f}% | "
            f"{s['mean_containment']*100:>7.2f}% | "
            f"{s['mean_bbox_iou']*100:>7.2f}% | "
            f"{s['mean_latency_ms']:>6.2f}ms"
        )
        print(row)
    print("=" * 95 + "\n")

    # Save summary markdown table
    md_summary_path = out_dir / "benchmark_summary.md"
    with open(md_summary_path, "w") as f:
        f.write("# YOLO Single Silkworm Benchmark Results (498 Test Samples)\n\n")
        f.write("| Model | Accuracy | Macro F1 | Healthy F1 | Grasserie F1 | BBox Containment | BBox IoU | Latency |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for name, data in results.items():
            s = data["summary"]
            f.write(
                f"| **{name}** | {s['accuracy']*100:.2f}% | {s['macro_f1']*100:.2f}% | "
                f"{s['healthy']['f1']*100:.2f}% | {s['grasserie']['f1']*100:.2f}% | "
                f"{s['mean_containment']*100:.2f}% | {s['mean_bbox_iou']*100:.2f}% | {s['mean_latency_ms']:.2f} ms |\n"
            )
    print(f"Saved benchmark summary markdown to {md_summary_path}")

    # Plot Confusion Matrices
    cm_path = str(out_dir / "confusion_matrices_comparison.png")
    plot_confusion_matrices(results, cm_path)


if __name__ == "__main__":
    main()
