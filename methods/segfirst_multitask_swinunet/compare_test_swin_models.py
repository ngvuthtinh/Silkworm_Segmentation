#!/usr/bin/env python3
"""
compare_test_swin_models.py — Comprehensive Test & Evaluation for 2 Swin-Unet Models
====================================================================================
Model 1: Joint Multi-Task Swin-Unet (Swin-Unet/multitask_silkworm_out/best_model.pth)
Model 2: SegFirst Multi-Task Swin-Unet (methods/segfirst_multitask_swinunet/results/.../phase2_best.pth)

Evaluates on data/silkworm_mixed_dataset/test (169 images):
  - Segmentation: Dice Score, mIoU, Coverage
  - Classification: Accuracy, Healthy vs Grasserie predictions
  - Side-by-side 5-panel visualizer for test samples
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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

CLASS_NAMES = {0: "Healthy", 1: "Grasserie (Diseased)"}
CLASS_COLORS = {0: (46, 204, 113), 1: (231, 76, 60)}  # Green, Red


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


def get_mask_prob(mask_logits: torch.Tensor, num_seg_classes: int) -> np.ndarray:
    if num_seg_classes == 2:
        probs = torch.softmax(mask_logits, dim=1)[:, 1, :, :]
    else:
        probs = torch.sigmoid(mask_logits)
    return probs.squeeze().cpu().numpy()


def evaluate_model_on_test_set(
    model: torch.nn.Module,
    num_seg_classes: int,
    test_samples: list[tuple[Path, Path, int]],
    device: torch.device,
    img_size: int = 224,
) -> dict:
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dices, ious = [], []
    cls_correct = 0
    cls_total = 0
    infer_times = []

    for img_p, mask_p, cls_label in test_samples:
        img_pil = Image.open(img_p).convert("RGB")
        mask_pil = Image.open(mask_p).convert("L")
        orig_w, orig_h = img_pil.size

        img_resized = TF.resize(img_pil, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = norm(TF.to_tensor(img_resized)).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            mask_logits, class_logits = model(tensor, phase=2)
        infer_times.append(time.time() - t0)

        # Segmentation metrics on original resolution
        prob = get_mask_prob(mask_logits, num_seg_classes)
        prob_full = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        ) / 255.0
        pred_bin = (prob_full >= 0.5).astype(np.uint8)
        gt_bin = (np.array(mask_pil) > 127).astype(np.uint8)

        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum() - intersection
        total_pixels = pred_bin.sum() + gt_bin.sum()

        dice = (2.0 * intersection + 1e-6) / (total_pixels + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)
        dices.append(dice)
        ious.append(iou)

        # Classification metrics
        if cls_label >= 0 and class_logits is not None:
            pred_class = int(torch.argmax(class_logits, dim=1).item())
            if pred_class == cls_label:
                cls_correct += 1
            cls_total += 1

    return {
        "mean_dice": float(np.mean(dices)) * 100.0,
        "mean_iou": float(np.mean(ious)) * 100.0,
        "cls_acc": (cls_correct / cls_total * 100.0) if cls_total > 0 else 0.0,
        "cls_correct": cls_correct,
        "cls_total": cls_total,
        "avg_latency_ms": float(np.mean(infer_times)) * 1000.0,
        "fps": 1.0 / max(1e-6, float(np.mean(infer_times))),
    }


def generate_side_by_side_comparison(
    model1: torch.nn.Module,
    num_seg_classes1: int,
    model2: torch.nn.Module,
    num_seg_classes2: int,
    samples_to_vis: list[tuple[Path, Path, int]],
    output_dir: Path,
    device: torch.device,
    img_size: int = 224,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for idx, (img_p, mask_p, cls_label) in enumerate(samples_to_vis, 1):
        img_pil = Image.open(img_p).convert("RGB")
        mask_pil = Image.open(mask_p).convert("L")
        orig_w, orig_h = img_pil.size
        img_np = np.array(img_pil)
        gt_bin = (np.array(mask_pil) > 127).astype(np.uint8)

        img_resized = TF.resize(img_pil, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = norm(TF.to_tensor(img_resized)).unsqueeze(0).to(device)

        with torch.no_grad():
            m1_mask, m1_cls = model1(tensor, phase=2)
            m1_prob = get_mask_prob(m1_mask, num_seg_classes1)
            m1_cls_probs = torch.softmax(m1_cls, dim=1).squeeze().cpu().numpy() if m1_cls is not None else [0.5, 0.5]

            m2_mask, m2_cls = model2(tensor, phase=2)
            m2_prob = get_mask_prob(m2_mask, num_seg_classes2)
            m2_cls_probs = torch.softmax(m2_cls, dim=1).squeeze().cpu().numpy() if m2_cls is not None else [0.5, 0.5]

        # Resize masks back
        m1_bin = (np.array(Image.fromarray((m1_prob * 255).astype(np.uint8)).resize((orig_w, orig_h))) >= 127).astype(np.uint8)
        m2_bin = (np.array(Image.fromarray((m2_prob * 255).astype(np.uint8)).resize((orig_w, orig_h))) >= 127).astype(np.uint8)

        # Overlays
        ov1 = img_np.copy()
        for c in range(3):
            ov1[:, :, c] = np.where(m1_bin == 1, ov1[:, :, c] * 0.45 + [220, 50, 255][c] * 0.55, ov1[:, :, c])

        ov2 = img_np.copy()
        for c in range(3):
            ov2[:, :, c] = np.where(m2_bin == 1, ov2[:, :, c] * 0.45 + [50, 200, 255][c] * 0.55, ov2[:, :, c])

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        # Panel 1: Original Image + GT label
        gt_text = CLASS_NAMES.get(cls_label, "Unlabeled") if cls_label >= 0 else "Multi-Silkworm"
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input: {img_p.name[:20]}\nGT: {gt_text}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground Truth Mask
        axes[1].imshow(gt_bin, cmap="gray")
        axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Model 1 Overlay
        m1_pred = int(np.argmax(m1_cls_probs))
        m1_conf = float(m1_cls_probs[m1_pred])
        m1_dice = (2.0 * (m1_bin * gt_bin).sum() + 1e-6) / (m1_bin.sum() + gt_bin.sum() + 1e-6)
        axes[2].imshow(ov1)
        axes[2].set_title(
            f"1️⃣ Joint Multi-Task Swin\nDice: {m1_dice:.1%} | Pred: {CLASS_NAMES[m1_pred]} ({m1_conf:.0%})",
            fontsize=11,
            color="purple",
            fontweight="bold",
        )
        axes[2].axis("off")

        # Panel 4: Model 2 Overlay
        m2_pred = int(np.argmax(m2_cls_probs))
        m2_conf = float(m2_cls_probs[m2_pred])
        m2_dice = (2.0 * (m2_bin * gt_bin).sum() + 1e-6) / (m2_bin.sum() + gt_bin.sum() + 1e-6)
        axes[3].imshow(ov2)
        axes[3].set_title(
            f"2️⃣ SegFirst Multi-Task Swin\nDice: {m2_dice:.1%} | Pred: {CLASS_NAMES[m2_pred]} ({m2_conf:.0%})",
            fontsize=11,
            color="#0077b6",
            fontweight="bold",
        )
        axes[3].axis("off")

        # Panel 5: Mask Difference
        diff_canvas = np.zeros_like(img_np)
        both = (m1_bin == 1) & (m2_bin == 1)
        m1_only = (m1_bin == 1) & (m2_bin == 0)
        m2_only = (m1_bin == 0) & (m2_bin == 1)
        diff_canvas[both] = [46, 204, 113]      # Green: Both agree
        diff_canvas[m1_only] = [231, 76, 60]    # Red: M1 only
        diff_canvas[m2_only] = [52, 152, 219]   # Blue: M2 only
        axes[4].imshow(diff_canvas)
        axes[4].set_title("Mask Comparison\nGreen=Both | Red=M1 | Blue=M2", fontsize=11, fontweight="bold")
        axes[4].axis("off")

        plt.tight_layout()
        save_file = output_dir / f"compare_{idx:02d}_{img_p.stem}.png"
        plt.savefig(str(save_file), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  [{idx:02d}/{len(samples_to_vis)}] Generated: {save_file.name} (M1 Dice={m1_dice:.1%}, M2 Dice={m2_dice:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Compare 2 Swin-Unet models on Test set")
    parser.add_argument("--ckpt1", type=str, default="Swin-Unet/multitask_silkworm_out/best_model.pth")
    parser.add_argument(
        "--ckpt2",
        type=str,
        default="methods/segfirst_multitask_swinunet/results/run_20260917_025501/checkpoints/phase2_best.pth",
    )
    parser.add_argument("--test_root", type=str, default="data/silkworm_mixed_dataset/test")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="methods/segfirst_multitask_swinunet/results/test_comparison",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_vis", type=int, default=12)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" 📊 BENCHMARK SO SÁNH 2 MÔ HÌNH SWIN-UNET TRÊN TẬP TEST")
    print("=" * 80)
    print(f" Model 1 (Joint Multi-Task)   : {args.ckpt1}")
    print(f" Model 2 (SegFirst Multi-Task): {args.ckpt2}")
    print(f" Test Dataset                 : {args.test_root}")
    print(f" Device                       : {device}")
    print("=" * 80)

    # 1. Gather test samples
    img_dir = Path(args.test_root) / "images"
    mask_dir = Path(args.test_root) / "masks"

    all_test_samples: list[tuple[Path, Path, int]] = []
    for img_p in sorted(img_dir.iterdir()):
        if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        mask_p = mask_dir / f"{img_p.stem}.png"
        if not mask_p.exists():
            mask_p = mask_dir / f"{img_p.stem}.jpg"
        if not mask_p.exists():
            continue

        name_lower = img_p.name.lower()
        if "grasserie" in name_lower:
            cls_id = 1
        elif "healthy" in name_lower:
            cls_id = 0
        else:
            cls_id = -1

        all_test_samples.append((img_p, mask_p, cls_id))

    print(f"Loaded {len(all_test_samples)} test samples from {args.test_root}\n")

    # 2. Load models
    print("Loading Model 1 (Joint Multi-Task Swin-Unet)...")
    m1, num_cls1 = load_model(args.ckpt1, device=device)
    print(f"  Model 1 loaded! (Seg output classes = {num_cls1})")

    print("Loading Model 2 (SegFirst Multi-Task Swin-Unet)...")
    m2, num_cls2 = load_model(args.ckpt2, device=device)
    print(f"  Model 2 loaded! (Seg output classes = {num_cls2})")

    # 3. Quantitative Evaluation
    print("\n--- [1/2] Đang đánh giá định lượng trên toàn bộ 169 ảnh tập Test ---")
    print("Evaluating Model 1 (Joint Multi-Task)...")
    res1 = evaluate_model_on_test_set(m1, num_cls1, all_test_samples, device=device)

    print("Evaluating Model 2 (SegFirst Multi-Task)...")
    res2 = evaluate_model_on_test_set(m2, num_cls2, all_test_samples, device=device)

    # Print Comparison Table
    print("\n" + "=" * 85)
    print(" 🏆 KẾT QUẢ SO SÁNH ĐỊNH LƯỢNG TRÊN TẬP TEST (169 MẪU)")
    print("=" * 85)
    header = f"{'Metric / Chỉ số':<30} | {'Model 1 (Joint Multi-Task)':<25} | {'Model 2 (SegFirst Multi-Task)':<25}"
    print(header)
    print("-" * 85)
    print(f"{'Silkworm Dice Score (%)':<30} | {res1['mean_dice']:<25.2f} | {res2['mean_dice']:<25.2f}")
    print(f"{'Silkworm mIoU (%)':<30} | {res1['mean_iou']:<25.2f} | {res2['mean_iou']:<25.2f}")
    print(
        f"{'Disease Accuracy (%)':<30} | {res1['cls_acc']:<25.2f} ({res1['cls_correct']}/{res1['cls_total']}) | {res2['cls_acc']:<25.2f} ({res2['cls_correct']}/{res2['cls_total']})"
    )
    print(f"{'Inference Latency (ms)':<30} | {res1['avg_latency_ms']:<25.2f} | {res2['avg_latency_ms']:<25.2f}")
    print(f"{'Throughput (FPS)':<30} | {res1['fps']:<25.1f} | {res2['fps']:<25.1f}")
    print("=" * 85)

    # 4. Generate Visual Comparisons
    print(f"\n--- [2/2] Đang tạo {args.num_vis} ảnh trực quan hóa đối đầu 5 khung hình ---")
    healthy_samples = [s for s in all_test_samples if s[2] == 0][:4]
    grasserie_samples = [s for s in all_test_samples if s[2] == 1][:4]
    tray_samples = [s for s in all_test_samples if s[2] == -1][:4]
    vis_samples = healthy_samples + grasserie_samples + tray_samples

    generate_side_by_side_comparison(
        model1=m1,
        num_seg_classes1=num_cls1,
        model2=m2,
        num_seg_classes2=num_cls2,
        samples_to_vis=vis_samples,
        output_dir=out_path / "visualizations",
        device=device,
    )

    print(f"\n🎉 Toàn bộ ảnh trực quan hóa đã được lưu tại: {out_path / 'visualizations'}")


if __name__ == "__main__":
    main()
