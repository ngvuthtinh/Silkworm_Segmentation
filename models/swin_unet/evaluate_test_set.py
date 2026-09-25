"""
evaluate_test_set.py — Comprehensive evaluation and visual inference for Multi-Task Swin-Unet
on the 169 silkworm test set images.
Computes:
  - mIoU, Dice Score, Pixel Accuracy, Precision, Recall
  - Classification Accuracy (Healthy vs Grasserie)
  - Produces 5 side-by-side visualization images
"""

from __future__ import annotations

import os
import sys
import glob
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import get_config
from train_multitask_silkworm import MultiTaskSwinUnet

CLASS_NAMES = {0: "Healthy", 1: "Grasserie (Diseased)"}


def evaluate(
    checkpoint_path: str = "runs/swin_unet_baseline/multitask_silkworm_out/best_model.pth",
    test_dir: str = "data/silkworm_mixed_dataset/test",
    output_dir: str = "runs/multitask_swinunet/inference_results",
    artifact_dir: str = "/home/subnh5/.gemini/antigravity-ide/brain/a99ebf88-1269-4695-bbe0-42c6ada4a9ee",
    cfg_path: str = "models/swin_unet/configs/swin_tiny_patch4_window7_224_lite.yaml",
    img_size: int = 224,
    gpu_id: str = "1",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    print("=" * 70)
    print(" 🐛 Multi-Task Swin-Unet Test Set Evaluation & Visualization")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Test Dir  : {test_dir}")
    print(f" Device    : {device}")
    print("=" * 70)

    # 1. Build Model
    class Args:
        cfg = cfg_path
        dataset = "silkworm_multitask"
        batch_size = 1
        zip = False
        cache_mode = "part"
        resume = None
        accumulation_steps = None
        use_checkpoint = False
        amp_opt_level = "O1"
        tag = None
        eval = True
        throughput = False
        opts = None

    swin_cfg = get_config(Args())
    model = MultiTaskSwinUnet(
        config=swin_cfg,
        img_size=img_size,
        num_seg_classes=2,
        num_cls_classes=2,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 2. Gather Test Data
    test_path = Path(test_dir)
    img_dir = test_path / "images"
    mask_dir = test_path / "masks_id"
    if not mask_dir.exists():
        mask_dir = test_path / "masks"

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Found {len(img_files)} test samples.")

    # Accumulators
    dices = []
    ious = []
    precisions = []
    recalls = []
    pixel_accs = []
    cls_correct = 0
    total_samples = 0

    visual_samples_saved = 0

    for idx, img_p in enumerate(tqdm(img_files, desc="Evaluating Test Set")):
        mask_p = mask_dir / f"{img_p.stem}.png"
        if not mask_p.exists():
            mask_p = mask_dir / f"{img_p.stem}.jpg"

        orig_img = Image.open(img_p).convert("RGB")
        resized_img = orig_img.resize((img_size, img_size), Image.BILINEAR)

        gt_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        if mask_p.exists():
            gt_pil = Image.open(mask_p).convert("L").resize((img_size, img_size), Image.NEAREST)
            raw_gt = np.array(gt_pil)
            gt_mask = (raw_gt > 0).astype(np.uint8)

        # Ground truth class label from filename (Healthy: 0, Diseased: 1)
        gt_cls = 0 if "healthy" in img_p.name.lower() else 1

        img_t = norm(TF.to_tensor(resized_img)).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                seg_logits, cls_logits = model(img_t)

        # Segmentation prediction: argmax over class 0 (bg) vs 1 (silkworm)
        pred_mask = torch.argmax(F.softmax(seg_logits, dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)

        # Classification prediction
        cls_probs = F.softmax(cls_logits, dim=1)[0]
        cls_pred = int(torch.argmax(cls_probs).item())
        cls_conf = float(cls_probs[cls_pred].item())

        if cls_pred == gt_cls:
            cls_correct += 1
        total_samples += 1

        # Metrics for binary segmentation (foreground: silkworm = 1)
        tp = np.sum((pred_mask == 1) & (gt_mask == 1))
        fp = np.sum((pred_mask == 1) & (gt_mask == 0))
        fn = np.sum((pred_mask == 0) & (gt_mask == 1))
        tn = np.sum((pred_mask == 0) & (gt_mask == 0))

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 1.0
        p_acc = (tp + tn) / (tp + fp + fn + tn)

        dices.append(dice)
        ious.append(iou)
        precisions.append(prec)
        recalls.append(rec)
        pixel_accs.append(p_acc)

        # Save 5 visual sample images
        if visual_samples_saved < 5:
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            # 1. Original Image
            axes[0].imshow(resized_img)
            axes[0].set_title(f"1. Ảnh Gốc ({img_p.name})", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            # 2. Ground Truth Mask
            axes[1].imshow(resized_img)
            axes[1].imshow(gt_mask, cmap="Reds", alpha=0.5)
            axes[1].set_title("2. Ground Truth Mask (Đỏ)", fontsize=11, fontweight="bold")
            axes[1].axis("off")

            # 3. Predicted Mask Overlay
            axes[2].imshow(resized_img)
            axes[2].imshow(pred_mask, cmap="Purples", alpha=0.6)
            axes[2].set_title(f"3. Swin-Unet Mask (Dice: {dice*100:.1f}%)", fontsize=11, fontweight="bold")
            axes[2].axis("off")

            # 4. Multi-Task Classification Badge
            badge_color = "#27ae60" if cls_pred == 0 else "#e74c3c"
            axes[3].set_facecolor("#1e272e")
            axes[3].text(
                0.5, 0.55,
                f"Multi-Task Prediction:\n\n{CLASS_NAMES[cls_pred]}\n(Conf: {cls_conf*100:.1f}%)",
                fontsize=14, fontweight="bold", ha="center", va="center", color=badge_color,
                transform=axes[3].transAxes,
            )
            axes[3].text(
                0.5, 0.2,
                f"Ground Truth: {CLASS_NAMES[gt_cls]}",
                fontsize=11, ha="center", va="center", color="#ecf0f1",
                transform=axes[3].transAxes,
            )
            axes[3].set_title("4. Phân Loại Sức Khỏe Con Tằm", fontsize=11, fontweight="bold")
            axes[3].axis("off")

            plt.tight_layout()
            out_file = f"swin_multitask_sample_{visual_samples_saved+1}.png"
            save_path = os.path.join(output_dir, out_file)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            shutil.copy2(save_path, os.path.join(artifact_dir, out_file))
            visual_samples_saved += 1

    # Summary Metrics
    mean_dice = np.mean(dices) * 100.0
    mean_iou = np.mean(ious) * 100.0
    mean_prec = np.mean(precisions) * 100.0
    mean_rec = np.mean(recalls) * 100.0
    mean_pacc = np.mean(pixel_accs) * 100.0
    cls_acc = (cls_correct / total_samples) * 100.0

    print("\n" + "=" * 65)
    print(" 🏆 MULTI-TASK SWIN-UNET FINAL BENCHMARK RESULTS (169 Test Images):")
    print(f"   Silkworm Dice Score : {mean_dice:.2f}%")
    print(f"   Silkworm mIoU       : {mean_iou:.2f}%")
    print(f"   Pixel Accuracy      : {mean_pacc:.2f}%")
    print(f"   Precision           : {mean_prec:.2f}%")
    print(f"   Recall              : {mean_rec:.2f}%")
    print(f"   Classification Acc  : {cls_acc:.2f}% ({cls_correct}/{total_samples} samples)")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
