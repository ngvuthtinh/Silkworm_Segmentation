"""
inference.py — Inference & 4-Panel Visualization Pipeline for SegFirst Multi-Task Swin-Unet.

Loads a trained model checkpoint and outputs:
    1. Input image
    2. Predicted segmentation mask
    3. Purple segment overlay
    4. Health status classification prediction (Healthy vs Diseased + confidence %)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
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

CLASS_NAMES = {0: "Healthy", 1: "Diseased"}


def run_inference(
    model_path: str,
    img_paths: list[str],
    output_dir: str,
    img_size: int = 224,
    gpu_id: str = "0",
) -> None:
    if torch.cuda.is_available():
        gpu_idx = int(gpu_id) if gpu_id.isdigit() else 0
        if gpu_idx >= torch.cuda.device_count():
            gpu_idx = 0
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" 🌟 SegFirst Multi-Task Swin-Unet Inference Pipeline")
    print(f" Checkpoint: {model_path}")
    print(f" Processing {len(img_paths)} images...")
    print(f" Output Directory: {output_dir}")
    print("=" * 70)

    cfg = SegFirstSwinConfig(input_size=img_size)
    model = SegFirstSwinUnet(config=cfg, input_size=img_size, num_seg_classes=1, num_cls_classes=2).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for idx, img_p in enumerate(img_paths, 1):
        img_pil = Image.open(img_p).convert("RGB")
        orig_w, orig_h = img_pil.size

        # Preprocess
        img_resized = TF.resize(img_pil, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
        tensor = norm(TF.to_tensor(img_resized)).unsqueeze(0).to(device)

        with torch.no_grad():
            mask_logits, class_logits = model(tensor, phase=2)
            mask_pred = torch.sigmoid(mask_logits)

        # Process segmentation prediction
        prob = mask_pred.squeeze().cpu().numpy()
        prob_full = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
        ) / 255.0
        mask_bin = (prob_full >= 0.5).astype(np.uint8)

        # Process classification prediction
        if class_logits is not None:
            cls_probs = torch.softmax(class_logits, dim=1).squeeze().cpu().numpy()
            cls_pred = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_pred])
            cls_label = f"{CLASS_NAMES[cls_pred]}\n({cls_conf*100:.1f}%)"
        else:
            cls_pred = 0
            cls_label = "N/A"

        # Create 4-panel visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        img_np = np.array(img_pil)

        # Panel 1: Original Image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input: {Path(img_p).name[:20]}", fontsize=11)
        axes[0].axis("off")

        # Panel 2: Predicted Mask
        axes[1].imshow(mask_bin, cmap="gray")
        axes[1].set_title(f"Predicted Mask (Coverage={mask_bin.mean():.1%})", fontsize=11)
        axes[1].axis("off")

        # Panel 3: Purple Overlay
        overlay = img_np.copy()
        color_purple = np.array([255, 0, 255], dtype=np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bin == 1, overlay[:, :, c] * 0.5 + color_purple[c] * 0.5, overlay[:, :, c]
            )
        axes[2].imshow(overlay)
        axes[2].set_title("SegFirst Swin-Unet Segment", fontsize=11, color="purple", fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Health Status Classification Badge
        badge_color = "green" if cls_pred == 0 else "red"
        axes[3].text(
            0.5,
            0.5,
            f"Health Status:\n\n{cls_label}",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            color=badge_color,
            transform=axes[3].transAxes,
        )
        axes[3].set_title("Classification Prediction", fontsize=11)
        axes[3].axis("off")

        plt.tight_layout()
        save_p = os.path.join(output_dir, f"infer_{Path(img_p).stem}.png")
        plt.savefig(save_p, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  [{idx:02d}/{len(img_paths)}] {Path(img_p).name} -> Mask Coverage={mask_bin.mean():.1%} | Health={CLASS_NAMES[cls_pred]}")

    print(f"\n[INFO] Inference completed! Visualizations saved to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description="SegFirst Multi-Task Swin-Unet Inference")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    p.add_argument("--image-dir", type=str, default="data/silkworm_mixed_dataset/test/images")
    p.add_argument("--output-dir", type=str, default="methods/segfirst_multitask_swinunet/results/inference_output")
    p.add_argument("--pattern", type=str, default="*")
    p.add_argument("--max-images", type=int, default=10)
    p.add_argument("--gpu", type=str, default="0")
    args = p.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.image_dir, f"{args.pattern}.jpg"))) + sorted(
        glob.glob(os.path.join(args.image_dir, f"{args.pattern}.png"))
    )
    if args.max_images > 0 and len(imgs) > args.max_images:
        imgs = imgs[: args.max_images]

    run_inference(
        model_path=args.checkpoint,
        img_paths=imgs,
        output_dir=args.output_dir,
        gpu_id=args.gpu,
    )


if __name__ == "__main__":
    main()
