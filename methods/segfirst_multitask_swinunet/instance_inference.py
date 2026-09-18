"""
instance_inference.py — Per-Silkworm Instance Analysis & Classification Pipeline for Swin-Unet.

Pipeline:
1. Run Swin-Unet Segmentation on whole image -> Output mask
2. Extract individual silkworm instances using Connected Components / Contours
3. Crop each silkworm instance bounding box
4. Pass cropped silkworm image into Classification Head -> Classify (Healthy vs Diseased)
5. Draw instance-level Bounding Boxes, Mask Overlays, and Labels on final output image
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
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
CLASS_COLORS_RGB = {0: (46, 204, 113), 1: (231, 76, 60)}  # Green / Red


def analyze_instance_silkworms(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    img_size: int = 224,
    min_area: int = 400,
    device: torch.device = torch.device("cpu"),
):
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    img_np = np.array(img_pil)

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 1. Whole-image segmentation
    img_resized = TF.resize(img_pil, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
    tensor = norm(TF.to_tensor(img_resized)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mask_logits, _ = model(tensor, phase=1)
        mask_pred = torch.sigmoid(mask_logits)

    prob = mask_pred.squeeze().cpu().numpy()
    prob_full = np.array(
        Image.fromarray((prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    ) / 255.0
    mask_bin = (prob_full >= 0.5).astype(np.uint8)

    # 2. Extract Individual Silkworm Contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    vis_img = img_np.copy()
    overlay_img = img_np.copy()

    instance_results = []

    for idx, cnt in enumerate(valid_contours, 1):
        x, y, w, h = cv2.boundingRect(cnt)

        # Expand padding slightly (10%) for cropping
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(orig_w, x + w + pad_x)
        y2 = min(orig_h, y + h + pad_y)

        crop_pil = img_pil.crop((x1, y1, x2, y2))
        crop_resized = TF.resize(crop_pil, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
        crop_tensor = norm(TF.to_tensor(crop_resized)).unsqueeze(0).to(device)

        with torch.no_grad():
            _, cls_logits = model(crop_tensor, phase=2)
            cls_probs = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()
            cls_pred = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_pred])

        color = CLASS_COLORS_RGB[cls_pred]
        color_bgr = (color[2], color[1], color[0])

        # Draw bounding box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color_bgr, 3)

        # Label text
        label_text = f"#{idx} {CLASS_NAMES[cls_pred]} {cls_conf*100:.0f}%"
        (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis_img, (x, max(0, y - txt_h - 10)), (x + txt_w + 10, y), color_bgr, -1)
        cv2.putText(vis_img, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Overlay contour mask
        cv2.drawContours(overlay_img, [cnt], -1, color_bgr, -1)

        instance_results.append(
            {"id": idx, "bbox": (x, y, w, h), "class": CLASS_NAMES[cls_pred], "confidence": cls_conf}
        )

    # Blend overlay
    alpha = 0.4
    blended = cv2.addWeighted(overlay_img, alpha, vis_img, 1 - alpha, 0)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    return instance_results


def main():
    p = argparse.ArgumentParser(description="Per-Silkworm Instance Analysis with SegFirst Swin-Unet")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--image", type=str, required=True, help="Path to input image")
    p.add_argument("--output", type=str, default="methods/segfirst_multitask_swinunet/results/instance_output.png")
    p.add_argument("--gpu", type=str, default="0")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cfg = SegFirstSwinConfig()
    model = SegFirstSwinUnet(config=cfg, input_size=224, num_seg_classes=1, num_cls_classes=2).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)

    print(f"\nAnalyzing instance silkworms on: {args.image}")
    results = analyze_instance_silkworms(model, args.image, args.output, device=device)
    print(f"Detected {len(results)} silkworms! Result saved to: {args.output}\n")


if __name__ == "__main__":
    main()
