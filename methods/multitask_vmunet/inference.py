"""
inference.py — Run multi-task VM-UNet on a new image.

Usage (from project root Silkworm_Segmentation/):

    # Single image
    python -m methods.multitask_vmunet.inference \
        --image path/to/image.jpg \
        --checkpoint methods/multitask_vmunet/results/run_XXXX/checkpoints/best_seg.pth

    # Directory of images
    python -m methods.multitask_vmunet.inference \
        --image-dir data/Silkworm\ Diseases.v1i.yolo26/test/images \
        --checkpoint methods/multitask_vmunet/results/run_XXXX/checkpoints/best_seg.pth \
        --output-dir results/inference/

    # With ground-truth mask directory for metrics
    python -m methods.multitask_vmunet.inference \
        --image-dir ... --mask-dir ... --checkpoint ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as T

# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "models" / "vmunet"))

from methods.multitask_vmunet.model  import MultiTaskVMUNet  # noqa
from methods.multitask_vmunet.config import MultiTaskConfig   # noqa

# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "Healthy", 1: "Diseased"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}  # green / red


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_path: str, img_size: int) -> tuple[torch.Tensor, Image.Image]:
    """Load image, return (tensor [1,3,H,W], original PIL)."""
    img_pil = Image.open(image_path).convert("RGB")
    tf = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(img_pil).unsqueeze(0)  # [1, 3, H, W]
    return tensor, img_pil


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_single(
    model: MultiTaskVMUNet,
    image_path: str,
    img_size: int = 256,
    seg_threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Run model on one image.

    Returns
    -------
    dict with keys:
        img_orig      : PIL Image (original)
        mask_prob     : np.ndarray [H, W]  sigmoid probabilities
        mask_binary   : np.ndarray [H, W]  thresholded mask
        class_idx     : int (0=healthy, 1=diseased)
        class_name    : str
        confidence    : float (0-1)
    """
    tensor, img_orig = _preprocess(image_path, img_size)
    tensor = tensor.to(device)

    model.eval()
    mask_pred, cls_logits = model(tensor)

    # Segmentation
    mask_prob = mask_pred.squeeze().cpu().numpy()   # [H, W] in [0,1]
    # Resize back to original resolution
    orig_w, orig_h = img_orig.size
    mask_prob_full = np.array(
        Image.fromarray((mask_prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    ) / 255.0
    mask_binary = (mask_prob_full >= seg_threshold).astype(np.uint8)

    # Classification
    cls_probs = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
    class_idx  = int(cls_probs.argmax())
    confidence = float(cls_probs[class_idx])

    return {
        "img_orig":    img_orig,
        "mask_prob":   mask_prob_full,
        "mask_binary": mask_binary,
        "class_idx":   class_idx,
        "class_name":  CLASS_NAMES[class_idx],
        "confidence":  confidence,
        "cls_probs":   cls_probs,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(
    result: dict,
    save_path: str | None = None,
    gt_mask: np.ndarray | None = None,
) -> None:
    """
    Plot:  Original | Segmentation mask | Overlay + class label
    Optionally a 4th panel with GT mask + IoU/Dice if gt_mask is given.
    """
    img_np     = np.array(result["img_orig"])
    mask_bin   = result["mask_binary"]
    class_name = result["class_name"]
    conf       = result["confidence"]
    color      = CLASS_COLORS[result["class_idx"]]

    # Build overlay (coloured mask on top of image)
    overlay = img_np.astype(np.float32).copy()
    rgb = (
        (255, 100, 100) if result["class_idx"] == 1 else (100, 255, 100)
    )  # red=diseased / green=healthy
    for c, val in enumerate(rgb):
        overlay[:, :, c] = np.where(mask_bin == 1,
                                    overlay[:, :, c] * 0.5 + val * 0.5,
                                    overlay[:, :, c])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(mask_bin, cmap="gray")
    axes[1].set_title("Predicted Mask", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\n{class_name} ({conf:.1%})",
        fontsize=12,
        color=color,
        fontweight="bold",
    )
    axes[2].axis("off")

    if gt_mask is not None:
        # Compute IoU & Dice
        pred_b = mask_bin.flatten().astype(bool)
        gt_b   = (gt_mask > 0).flatten()
        inter  = (pred_b & gt_b).sum()
        union  = (pred_b | gt_b).sum()
        iou    = (inter + 1e-6) / (union + 1e-6)
        dice   = (2 * inter + 1e-6) / (pred_b.sum() + gt_b.sum() + 1e-6)

        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title(
            f"Ground Truth\nIoU={iou:.3f}  Dice={dice:.3f}",
            fontsize=12,
        )
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Load model helper
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    config: MultiTaskConfig,
    device: torch.device,
) -> MultiTaskVMUNet:
    model = MultiTaskVMUNet(
        input_channels=3,
        num_seg_classes=config.num_seg_classes,
        num_cls_classes=config.num_cls_classes,
        depths=config.depths,
        depths_decoder=config.depths_decoder,
        drop_path_rate=config.drop_path_rate,
        cls_hidden=config.cls_hidden,
        cls_dropout=config.cls_dropout,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and checkpoint dict
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[INFO] Model loaded from: {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Task VM-UNet Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",     type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Directory of images for batch inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--mask-dir",   type=str, default=None,
                        help="Optional GT mask directory (for metric reporting)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualisation images")
    parser.add_argument("--img-size",   type=int, default=256)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--gpu",        type=str, default="0")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0 = all)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpu_idx = int(args.gpu) if args.gpu.isdigit() else 0
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    config = MultiTaskConfig(input_size=args.img_size)
    model  = load_model(args.checkpoint, config, device)

    # Collect image paths
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    if args.image:
        img_paths = [Path(args.image)]
    else:
        img_paths = sorted(
            p for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        if args.max_images > 0:
            img_paths = img_paths[:args.max_images]

    save_dir = Path(args.output_dir) if args.output_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Metrics accumulators
    all_ious, all_dices = [], []
    cls_preds_all, cls_gts_all = [], []

    print(f"[INFO] Running inference on {len(img_paths)} image(s) …")

    for i, img_path in enumerate(img_paths):
        result = infer_single(
            model=model,
            image_path=str(img_path),
            img_size=args.img_size,
            seg_threshold=args.threshold,
            device=device,
        )

        # Load GT mask if available
        gt_mask = None
        if args.mask_dir:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = Path(args.mask_dir) / (img_path.stem + ext)
                if candidate.exists():
                    gt_arr  = np.array(Image.open(candidate).convert("L"))
                    gt_mask = (gt_arr > 0).astype(np.uint8)
                    break
            if gt_mask is not None:
                pred_b = result["mask_binary"].flatten().astype(bool)
                gt_b   = gt_mask.flatten().astype(bool)
                inter  = (pred_b & gt_b).sum()
                union  = (pred_b | gt_b).sum()
                all_ious.append((inter + 1e-6) / (union + 1e-6))
                all_dices.append((2 * inter + 1e-6) / (pred_b.sum() + gt_b.sum() + 1e-6))

        # Save or show
        if save_dir:
            save_path = str(save_dir / f"pred_{img_path.stem}.png")
            visualise(result, save_path=save_path, gt_mask=gt_mask)
        else:
            visualise(result, gt_mask=gt_mask)

        print(
            f"  [{i+1}/{len(img_paths)}] {img_path.name} "
            f"→ {result['class_name']} ({result['confidence']:.1%})  "
            f"mask_coverage={result['mask_binary'].mean():.1%}"
        )

    # Summary metrics
    if all_ious:
        print(f"\n[SUMMARY] Segmentation over {len(all_ious)} images:")
        print(f"  Mean IoU  = {np.mean(all_ious):.4f}")
        print(f"  Mean Dice = {np.mean(all_dices):.4f}")

    if save_dir:
        print(f"\n[INFO] Visualisations saved to: {save_dir}")


if __name__ == "__main__":
    main()
