"""
inference.py — Visual inference and qualitative visualization for ViT-P on test images.
Generates side-by-side comparisons:
[Original Image | Ground Truth Mask | ViT-P Point Predictions | Predicted Mask | Overlay]
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vit_p.config import ViTPConfig
from experiments.vit_p.model import ViTPSegClassifier


def run_inference(
    checkpoint_path: str = "runs/vit_p/checkpoints/best_model.pth",
    num_samples: int = 5,
    output_dir: str = "runs/vit_p/inference_results",
    artifact_dir: str = "/home/subnh5/.gemini/antigravity-ide/brain/a99ebf88-1269-4695-bbe0-42c6ada4a9ee",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading ViT-P model from {checkpoint_path} on {device}...")

    # Load model
    model = ViTPSegClassifier(
        arch="dinov2_vits14",
        num_classes=2,
        num_points=256,   # Dense 16x16 grid for high-resolution point evaluation
        img_size=224,
        patch_size=14,
        pretrained=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Adapt state dict if head/backbone points shape is flexible
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_img_dir = Path("data/silkworm_mixed_dataset/test/images")
    test_mask_dir = Path("data/silkworm_mixed_dataset/test/masks_id")
    if not test_mask_dir.exists():
        test_mask_dir = Path("data/silkworm_mixed_dataset/test/masks")

    valid_imgs = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    selected_imgs = valid_imgs[:num_samples]

    # Create uniform grid of 16x16 = 256 points in [-1, 1]
    grid_y, grid_x = np.meshgrid(np.linspace(-0.9, 0.9, 16), np.linspace(-0.9, 0.9, 16))
    grid_points = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1).astype(np.float32)  # (256, 2)
    grid_tensor = torch.from_numpy(grid_points).unsqueeze(0).to(device)                   # (1, 256, 2)

    generated_files = []

    for i, img_p in enumerate(selected_imgs):
        mask_p = test_mask_dir / f"{img_p.stem}.png"
        if not mask_p.exists():
            mask_p = test_mask_dir / f"{img_p.stem}.jpg"

        orig_pil = Image.open(img_p).convert("RGB")
        orig_resized = orig_pil.resize((224, 224), Image.BILINEAR)

        gt_mask = np.zeros((224, 224), dtype=np.uint8)
        if mask_p.exists():
            gt_pil = Image.open(mask_p).convert("L").resize((224, 224), Image.NEAREST)
            raw_gt = np.array(gt_pil)
            gt_mask = (raw_gt > 0).astype(np.uint8)

        img_t = TF.to_tensor(orig_resized)
        norm_img_t = normalize(img_t).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(norm_img_t, grid_tensor)  # (1, 256, 2)
                probs = F.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()  # Prob of silkworm (256,)

        # Reshape point probabilities to 16x16 grid and interpolate to 224x224 dense mask
        prob_grid = probs.reshape(1, 1, 16, 16)
        prob_dense = F.interpolate(torch.from_numpy(prob_grid).float(), size=(224, 224), mode="bicubic", align_corners=True)[0, 0].numpy()
        pred_mask = (prob_dense > 0.5).astype(np.uint8)

        # Plot 4-panel visual comparison
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        
        # 1. Original Image
        axes[0].imshow(orig_resized)
        axes[0].set_title(f"1. Ảnh Gốc ({img_p.name})", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground Truth
        axes[1].imshow(orig_resized)
        axes[1].imshow(gt_mask, cmap="Reds", alpha=0.5)
        axes[1].set_title("2. Ground Truth Mask (Thực tế)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # 3. Point Predictions
        axes[2].imshow(orig_resized)
        # Convert [-1, 1] coords to [0, 224] pixel coords
        px = ((grid_points[:, 1] + 1) / 2.0) * 224
        py = ((grid_points[:, 0] + 1) / 2.0) * 224
        for pt_x, pt_y, pr in zip(px, py, probs):
            color = "lime" if pr > 0.5 else "red"
            alpha = 0.8 if pr > 0.5 else 0.4
            axes[2].scatter(pt_x, pt_y, c=color, s=20, alpha=alpha, edgecolors="none")
        axes[2].set_title("3. ViT-P Points (Xanh: Tằm, Đỏ: Nền)", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        # 4. Dense Predicted Overlay
        axes[3].imshow(orig_resized)
        axes[3].imshow(pred_mask, cmap="Greens", alpha=0.55)
        axes[3].set_title("4. Phân Đoạn Dự Đoán (ViT-P Overlay)", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        out_filename = f"vit_p_prediction_sample_{i+1}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Copy to artifact dir for direct Markdown rendering
        artifact_path = os.path.join(artifact_dir, out_filename)
        shutil.copy2(out_path, artifact_path)
        generated_files.append(artifact_path)
        print(f"Generated sample visualization: {out_filename}")

    print(f"\nSuccessfully generated {len(generated_files)} visual prediction images!")
    return generated_files


if __name__ == "__main__":
    run_inference()
