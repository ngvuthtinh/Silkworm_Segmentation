"""
instance_inference.py — Per-Silkworm Instance Analysis & Classification Pipeline

Pipeline:
1. Run VM-UNet Segmentation on whole image -> Output mask
2. Extract individual silkworm instances using Connected Components / Contours
3. Crop each silkworm instance bounding box
4. Pass cropped silkworm image into Classification Head -> Classify (Healthy vs Diseased)
5. Draw instance-level Bounding Boxes, Mask Overlays, and Labels on final output image
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Insert project path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))

from model import SegFirstVMUNet

CLASS_NAMES = {0: "Healthy", 1: "Diseased"}
CLASS_COLORS_RGB = {0: (46, 204, 113), 1: (231, 76, 60)} # Green / Red

def analyze_instance_silkworms(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    img_size: int = 256,
    min_area: int = 400,
    device: torch.device = torch.device("cpu")
):
    # Load Image
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    img_np = np.array(img_pil)

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 1. Whole-image segmentation
    img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    tensor = norm(torch.from_numpy(img_arr.transpose(2, 0, 1))).unsqueeze(0).to(device)

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

    # Canvas for visualization
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

        # Preprocess crop for Classification Head
        crop_resized = crop_pil.resize((img_size, img_size), Image.BILINEAR)
        crop_arr = np.array(crop_resized, dtype=np.float32) / 255.0
        crop_tensor = norm(torch.from_numpy(crop_arr.transpose(2, 0, 1))).unsqueeze(0).to(device)

        with torch.no_grad():
            _, crop_cls_logits = model(crop_tensor, phase=2)
            cls_probs = torch.softmax(crop_cls_logits, dim=1).squeeze().cpu().numpy()
            cls_pred = int(np.argmax(cls_probs))
            cls_conf = float(cls_probs[cls_pred])

        cls_label = CLASS_NAMES[cls_pred]
        color_rgb = CLASS_COLORS_RGB[cls_pred]

        instance_results.append({
            "id": idx,
            "bbox": (x1, y1, x2, y2),
            "class": cls_label,
            "confidence": cls_conf,
            "area": cv2.contourArea(cnt)
        })

        # Draw contour mask on overlay
        cnt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
        for c_idx in range(3):
            overlay_img[:, :, c_idx] = np.where(
                cnt_mask == 1,
                overlay_img[:, :, c_idx] * 0.4 + color_rgb[c_idx] * 0.6,
                overlay_img[:, :, c_idx]
            )

        # Draw Bounding Box and Label on vis_img
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color_rgb, 3)

        label_text = f"Tằm #{idx}: {cls_label} ({cls_conf*100:.1f}%)"
        (t_w, t_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, max(0, y1 - t_h - 10)), (x1 + t_w + 10, max(0, y1)), color_rgb, -1)
        cv2.putText(
            vis_img,
            label_text,
            (x1 + 5, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Create 3-panel visualization plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_np)
    axes[0].set_title(f"Input: {Path(image_path).name}\n({len(valid_contours)} tằm detected)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title("Instance Segmentation Overlay\n(Xanh=Khỏe | Đỏ=Bệnh)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(vis_img)
    axes[2].set_title("Per-Silkworm Disease Diagnosis", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return instance_results


def main():
    p = argparse.ArgumentParser(description="Per-Silkworm Instance Analysis")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image-dir", type=str, default="data/test")
    p.add_argument("--output-dir", type=str, default="methods/segfirst_multitask_vmunet/results/instance_analysis")
    p.add_argument("--gpu", type=str, default="1")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_idx = int(args.gpu) if args.gpu.isdigit() else 0
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = SegFirstVMUNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png"))) + sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))

    print(f"============================================================")
    print(f" Per-Silkworm Instance Analysis & Classification Pipeline")
    print(f" Processing {len(img_paths)} images from {args.image_dir}...")
    print(f"============================================================")

    for idx, img_p in enumerate(img_paths, 1):
        out_p = os.path.join(args.output_dir, f"instance_{Path(img_p).stem}.png")
        results = analyze_instance_silkworms(model, img_p, out_p, device=device)

        print(f"\n📸 Ảnh [{idx}/{len(img_paths)}]: {Path(img_p).name}")
        print(f"   Phát hiện tổng cộng: {len(results)} con tằm trong ảnh.")
        for r in results:
            status_symbol = "🟢 Khỏe" if r['class'] == 'Healthy' else "🔴 Bệnh"
            print(f"   └─ Con tằm #{r['id']}: {status_symbol} ({r['class']}) - Độ tin cậy: {r['confidence']*100:.1f}% | Area: {r['area']}px")

    print(f"\n[INFO] Complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
