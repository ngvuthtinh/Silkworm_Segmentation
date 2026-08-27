import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config_setting import setting_config
from models.vmunet.vmunet import VMUNet

def find_best_checkpoint(results_dir):
    pattern = os.path.join(results_dir, "vmunet_silkworm_*", "checkpoints", "best-*.pth")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(results_dir, "**", "*.pth"), recursive=True)
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in " + results_dir)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

def main():
    parser = argparse.ArgumentParser(description="VM-UNet Silkworm Prediction & Visualization")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing images for prediction")
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory containing ground truth masks (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output visualizations")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum number of images to process")
    args = parser.parse_args()

    config = setting_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    checkpoint_path = find_best_checkpoint(results_dir)
    print(f"[INFO] Loading best model checkpoint: {checkpoint_path}")

    model = VMUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    cleaned_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()

    # Determine input directory
    if args.image_dir:
        val_img_dir = os.path.abspath(args.image_dir)
        val_msk_dir = os.path.abspath(args.mask_dir) if args.mask_dir else None
    else:
        # Default to root data silkworm val/images
        val_img_dir = os.path.join(config.data_path, "val", "images")
        val_msk_dir = os.path.join(config.data_path, "val", "masks")

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    img_files = [f for f in sorted(os.listdir(val_img_dir)) if f.lower().endswith(valid_exts)]
    
    if args.max_images > 0 and len(img_files) > args.max_images:
        img_files = img_files[:args.max_images]

    save_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Source Image Directory: {val_img_dir}")
    print(f"[INFO] Running inference on {len(img_files)} images...")

    for idx, fname in enumerate(img_files):
        img_path = os.path.join(val_img_dir, fname)
        orig_img_pil = Image.open(img_path).convert("RGB")
        orig_img_np = np.array(orig_img_pil)
        w_orig, h_orig = orig_img_pil.size

        # Check ground truth mask
        has_gt = False
        gt_msk_binary = None
        if val_msk_dir and os.path.exists(val_msk_dir):
            base_name = os.path.splitext(fname)[0]
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(val_msk_dir, base_name + ext)
                if os.path.exists(candidate):
                    gt_msk_np = np.array(Image.open(candidate).convert("L"))
                    gt_msk_binary = (gt_msk_np > 128).astype(np.uint8)
                    has_gt = True
                    break

        # Preprocess with test_transformer for exact model input
        dummy_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        img_tensor, _ = config.test_transformer((orig_img_np, dummy_mask))
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

        pred_pil = Image.fromarray((pred_prob * 255).astype(np.uint8)).resize((w_orig, h_orig), Image.BILINEAR)
        pred_prob_orig = np.array(pred_pil) / 255.0
        pred_binary = (pred_prob_orig > 0.5).astype(np.uint8)

        # Create Overlay: Green mask on silkworms
        green_mask = np.zeros_like(orig_img_np, dtype=np.float32)
        green_mask[pred_binary == 1] = [0, 255, 0]
        has_mask = (pred_binary == 1)[:, :, np.newaxis]
        overlay = np.where(has_mask, orig_img_np.astype(np.float32) * 0.6 + green_mask * 0.4, orig_img_np.astype(np.float32))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        if has_gt and gt_msk_binary is not None:
            intersection = np.logical_and(pred_binary, gt_msk_binary).sum()
            union = np.logical_or(pred_binary, gt_msk_binary).sum()
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2 * intersection + 1e-6) / (pred_binary.sum() + gt_msk_binary.sum() + 1e-6)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(orig_img_np)
            axes[0].set_title(f"Image: {fname}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(gt_msk_binary, cmap="gray")
            axes[1].set_title("Ground Truth Mask", fontsize=11)
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"VM-UNet Pred (IoU: {iou:.2%}, Dice: {dice:.2%})", fontsize=11)
            axes[2].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(orig_img_np)
            axes[0].set_title(f"Image: {fname}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title("VM-UNet Prediction Overlay", fontsize=11)
            axes[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"predict_{os.path.splitext(fname)[0]}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        print(f" [{idx+1}/{len(img_files)}] Saved prediction: {save_path}")

    print(f"\n[SUCCESS] Processed {len(img_files)} images. Visualizations saved in: {save_dir}")

if __name__ == "__main__":
    main()
