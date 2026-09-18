"""
train_and_predict_silkynet.py — PyTorch implementation of Silkynet (U-Net)
Trains U-Net on Silkynet dataset and runs test inference on both Silkynet and YOLO datasets.
"""

import os
import sys
import glob
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# 1. Silkynet U-Net Model Architecture (PyTorch)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SilkynetUNet(nn.Module):
    """Classic U-Net matching Silkynet specification."""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(ch, feature))
            ch = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final Conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            concat_x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# 2. Dataset for Silkynet
# ---------------------------------------------------------------------------

class SilkynetSegDataset(Dataset):
    def __init__(self, silkynet_root: str, split: str = "train", img_size: int = 256, is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train
        self.samples = []

        root = Path(silkynet_root)

        # Check if root follows standard split format: root / split / images & masks
        split_img_dir = root / split / "images"
        split_mask_dir = root / split / "masks"

        if split_img_dir.exists():
            for img_p in sorted(split_img_dir.iterdir()):
                if img_p.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    stem = img_p.stem
                    mask_p = None
                    if split_mask_dir.exists():
                        for ext in [".png", ".jpg", ".jpeg"]:
                            cand = split_mask_dir / (stem + ext)
                            if cand.exists():
                                mask_p = cand
                                break
                    if mask_p or split == "test":
                        self.samples.append((img_p, mask_p))
        else:
            # Fallback to legacy silkynet layout
            pairs = [
                (root / "larvaTrain" / "img", root / "larvaTrain" / "label"),
                (root / "output20221127" / "JPEGImages", root / "output20221127" / "SegmentationClassPNG")
            ]
            for img_dir, mask_dir in pairs:
                if not img_dir.exists():
                    continue
                for img_p in sorted(img_dir.iterdir()):
                    if img_p.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                        continue
                    stem = img_p.stem
                    mask_p = None
                    if mask_dir.exists():
                        for ext in [".png", ".jpg", ".jpeg"]:
                            cand = mask_dir / (stem + ext)
                            if cand.exists():
                                mask_p = cand
                                break
                    if mask_p or split == "test":
                        self.samples.append((img_p, mask_p))

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))
        img_tensor = self.norm(img_tensor)

        if mask_p is not None and Path(mask_p).exists():
            mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
            mask_arr = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
        else:
            mask_arr = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_tensor, mask_tensor


# ---------------------------------------------------------------------------
# 3. Main Train & Predict
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    out_root = Path("methods/silkynet/results/silkynet_unet_test_output")
    out_silk = out_root / "silkynet_test"
    out_yolo = out_root / "yolo_test"
    out_root.mkdir(parents=True, exist_ok=True)
    out_silk.mkdir(parents=True, exist_ok=True)
    out_yolo.mkdir(parents=True, exist_ok=True)

    # 1. Train Silkynet U-Net
    print("[INFO] Loading Silkynet training dataset...")
    ds_train = SilkynetSegDataset("methods/silkynet/data", img_size=256, is_train=True)
    loader_train = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=2)
    print(f"[INFO] Training Silkynet U-Net on {len(ds_train)} image-mask pairs...")

    model = SilkynetUNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()

    epochs = 25
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader_train:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
        mean_loss = total_loss / len(ds_train)
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch [{ep:02d}/{epochs:02d}] Loss = {mean_loss:.4f}")

    ckpt_path = out_root / "silkynet_unet_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Silkynet U-Net trained in {time.time() - t0:.1f}s — weights saved to {ckpt_path}")

    # 2. Prediction Helper
    model.eval()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def predict_and_save(img_paths, mask_dir, save_dir, dataset_name):
        print(f"\n============================================================")
        print(f"[INFO] Silkynet U-Net Prediction on {dataset_name} ({len(img_paths)} images)...")
        print(f"============================================================")
        ious, dices = [], []

        for i, img_p in enumerate(img_paths, 1):
            img_pil = Image.open(img_p).convert("RGB")
            orig_w, orig_h = img_pil.size

            img_resized = img_pil.resize((256, 256), Image.BILINEAR)
            tensor = norm(torch.from_numpy(np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

            prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
            mask_bin = (prob_full >= 0.5).astype(np.uint8)

            gt_mask = None
            if mask_dir:
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = Path(mask_dir) / (Path(img_p).stem + ext)
                    if cand.exists():
                        gt_arr = np.array(Image.open(cand).convert("L"))
                        gt_mask = (gt_arr > 0).astype(np.uint8)
                        break

            n_panels = 4 if gt_mask is not None else 3
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

            img_np = np.array(img_pil)
            axes[0].imshow(img_np)
            axes[0].set_title(f"Input: {Path(img_p).name[:25]}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(mask_bin, cmap="gray")
            axes[1].set_title(f"Predicted Mask (cov={mask_bin.mean():.1%})", fontsize=11)
            axes[1].axis("off")

            overlay = img_np.copy()
            color_purple = np.array([255, 0, 255], dtype=np.float32)
            for c in range(3):
                overlay[:, :, c] = np.where(mask_bin == 1, overlay[:, :, c] * 0.5 + color_purple[c] * 0.5, overlay[:, :, c])
            axes[2].imshow(overlay)
            axes[2].set_title("Silkynet U-Net Segment", fontsize=11, color="purple", fontweight="bold")
            axes[2].axis("off")

            if gt_mask is not None:
                pred_b = mask_bin.flatten().astype(bool)
                gt_b = gt_mask.flatten().astype(bool)
                inter = (pred_b & gt_b).sum()
                union = (pred_b | gt_b).sum()
                iou = (inter + 1e-6) / (union + 1e-6)
                dice = (2 * inter + 1e-6) / (pred_b.sum() + gt_b.sum() + 1e-6)
                ious.append(iou)
                dices.append(dice)

                axes[3].imshow(gt_mask, cmap="gray")
                axes[3].set_title(f"GT Mask\nIoU={iou:.3f} Dice={dice:.3f}", fontsize=11)
                axes[3].axis("off")

            plt.tight_layout()
            save_p = str(save_dir / f"pred_{Path(img_p).stem}.png")
            plt.savefig(save_p, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  [{i:02d}/{len(img_paths)}] {Path(img_p).name[:30]} -> Mask Coverage={mask_bin.mean():.1%}")

        if ious:
            print(f"\n[SUMMARY] {dataset_name} Evaluation Metrics:")
            print(f"  Mean IoU  = {np.mean(ious):.4f} ({np.mean(ious)*100:.2f}%)")
            print(f"  Mean Dice = {np.mean(dices):.4f} ({np.mean(dices)*100:.2f}%)")

        print(f"[INFO] Visualisations saved to: {save_dir}")

    # 3. Run predictions on Silkynet Test Set
    silk_imgs = sorted(glob.glob("methods/silkynet/data/larvaTest/img/*.jpg"))[:10]
    predict_and_save(silk_imgs, "methods/silkynet/data/larvaTest/label", out_silk, "Silkynet Test Dataset")

    # 4. Run predictions on YOLO Test Set
    yolo_imgs = sorted(glob.glob("data/Silkworm Diseases.v1i.yolo26/test/images/*.jpg"))[:10]
    predict_and_save(yolo_imgs, None, out_yolo, "YOLO Test Dataset")

if __name__ == "__main__":
    main()
