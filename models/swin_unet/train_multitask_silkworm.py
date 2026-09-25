#!/usr/bin/env python3
"""
train_multitask_silkworm.py — Multi-Task Swin-Unet Training Pipeline
====================================================================
Architecture: Multi-Task Swin-Unet (Swin-T Backbone + UNet Decoder + Bottleneck MLP Classifier)
Tasks:
  1. Segmentation: Silkworm boundary & shape mask (Background vs Silkworm)
  2. Classification: Health diagnosis (0: Healthy vs 1: Grasserie / Diseased)
Dataset: data/silkworm_mixed_dataset
Loss: Loss_Seg (0.4*CE + 0.6*Dice) + 0.1 * Loss_Cls (CrossEntropy)
Terminal Logging: Real-time percentage (% completed) logged directly to terminal.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Add project paths
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import get_config
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from utils import DiceLoss


class MultiTaskSwinUnet(nn.Module):
    """
    Multi-Task Swin-Unet:
    - Backbone & Decoder: SwinTransformerSys for Segmentation (num_classes=2)
    - Classification Head: Bottleneck feature (dim=768) -> Global Pool -> MLP -> 2 classes (Healthy vs Grasserie)
    """

    def __init__(
        self,
        config,
        img_size: int = 224,
        num_seg_classes: int = 2,
        num_cls_classes: int = 2,
        cls_hidden: int = 128,
        cls_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=num_seg_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )

        bottleneck_dim = int(config.MODEL.SWIN.EMBED_DIM * 2 ** (len(config.MODEL.SWIN.DEPTHS) - 1))  # 768 for Swin-T
        self.cls_head = nn.Sequential(
            nn.Linear(bottleneck_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, num_cls_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. Encoder forward pass to bottleneck
        bottleneck, x_downsample = self.swin_unet.forward_features(x)  # [B, 49, 768]

        # 2. Classification from bottleneck global average pooling
        cls_feat = bottleneck.mean(dim=1)  # [B, 768]
        cls_logits = self.cls_head(cls_feat)  # [B, 2]

        # 3. Decoder forward pass to segmentation mask
        x_up = self.swin_unet.forward_up_features(bottleneck, x_downsample)
        seg_logits = self.swin_unet.up_x4(x_up)  # [B, 2, H, W]

        return seg_logits, cls_logits

    def load_from(self, config) -> None:
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"Loading pretrained backbone weights from: {pretrained_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = pretrained_dict["model"]

            model_dict = self.swin_unet.state_dict()
            full_dict = copy_pretrained_weights(model_dict, pretrained_dict)
            self.swin_unet.load_state_dict(full_dict, strict=False)
            print(" Pretrained weights loaded successfully into Swin-Unet backbone!\n")


def copy_pretrained_weights(model_dict, pretrained_dict):
    full_dict = model_dict.copy()
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                full_dict[k] = v
    return full_dict


class MultiTaskSilkwormDataset(Dataset):
    """
    Multi-Task Dataset:
    Returns (image, mask, class_label)
    class_label:
        0: Healthy
        1: Grasserie (Diseased)
       -1: Unknown / Tray image without single health diagnosis
    """

    def __init__(self, data_root: str, split: str = "train", img_size: int = 224) -> None:
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.is_train = split == "train"

        root = Path(data_root) / split
        img_dir = root / "images"
        mask_dir = root / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Missing images or masks directory in {root}")

        self.samples: List[Tuple[Path, Path, int]] = []
        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            stem = img_p.stem
            mask_p = mask_dir / f"{stem}.png"
            if not mask_p.exists():
                mask_p = mask_dir / f"{stem}.jpg"
            if not mask_p.exists():
                continue

            # Identify health status from filename
            name_lower = img_p.name.lower()
            if "grasserie" in name_lower:
                cls_label = 1  # Diseased
            elif "healthy" in name_lower:
                cls_label = 0  # Healthy
            else:
                cls_label = -1  # Unlabeled for health

            self.samples.append((img_p, mask_p, cls_label))

        if not self.samples:
            raise RuntimeError(f"No valid image-mask pairs found in {root}")

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_p, mask_p, cls_label = self.samples[idx]

        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")

        img = TF.resize(img, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

        img_tensor = self.norm(TF.to_tensor(img))
        mask_np = np.array(mask)
        mask_tensor = (torch.from_numpy(mask_np) > 127).long()

        return img_tensor, mask_tensor, cls_label


def compute_dice_score(pred_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> float:
    preds = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
    pred_fg = (preds == 1).float()
    target_fg = (targets == 1).float()
    intersection = (pred_fg * target_fg).sum()
    total = pred_fg.sum() + target_fg.sum()
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return float(dice.item())


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def train_multitask_silkworm(args) -> None:
    gpu_id = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "multitask_train.log"

    logger = logging.getLogger("MultiTask_SwinUnet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    def log_print(msg: str) -> None:
        print(msg, flush=True)
        logger.info(msg)

    log_print("=" * 85)
    log_print("   🐛 MULTI-TASK SWIN-UNET TRAINING — SILKWORM SEGMENTATION & DISEASE DETECTION")
    log_print("=" * 85)
    log_print(f" Dataset Root     : {args.data_dir}")
    log_print(f" Output Directory : {args.output_dir}")
    log_print(f" Batch Size       : {args.batch_size}")
    log_print(f" Image Size       : {args.img_size}x{args.img_size}")
    log_print(f" Max Epochs       : {args.max_epochs}")
    log_print(f" Base LR          : {args.base_lr}")
    log_print(f" Loss Weights     : Lambda_Seg = {args.lambda_seg}, Lambda_Cls = {args.lambda_cls}")
    log_print(f" Device           : {device} (GPU ID: {gpu_id})")
    log_print("=" * 85)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_dataset = MultiTaskSilkwormDataset(args.data_dir, split="train", img_size=args.img_size)
    val_dataset = MultiTaskSilkwormDataset(args.data_dir, split="valid", img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    log_print(f" Train: {len(train_dataset)} samples ({len(train_loader)} batches/epoch)")
    log_print(f" Valid: {len(val_dataset)} samples ({len(val_loader)} batches/epoch)")

    config = get_config(args)
    model = MultiTaskSwinUnet(
        config=config,
        img_size=args.img_size,
        num_seg_classes=2,
        num_cls_classes=2,
        cls_hidden=128,
        cls_dropout=0.3,
    ).to(device)

    model.load_from(config)

    # Losses
    ce_loss_seg = nn.CrossEntropyLoss()
    dice_loss_seg = DiceLoss(n_classes=2)
    ce_loss_cls = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    total_epochs = args.max_epochs
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch

    global_step = 0
    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_val_acc = 0.0
    start_train_time = time.time()

    log_print("🚀 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN MULTI-TASK...")
    log_print("=" * 85)

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        model.train()

        train_loss_total = 0.0
        train_seg_total = 0.0
        train_cls_total = 0.0
        train_cls_correct = 0
        train_cls_count = 0

        for step, (images, masks, cls_labels) in enumerate(train_loader, 1):
            global_step += 1
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)

            seg_logits, cls_logits = model(images)

            # 1. Segmentation Loss
            l_ce = ce_loss_seg(seg_logits, masks)
            l_dice = dice_loss_seg(seg_logits, masks, softmax=True)
            l_seg = 0.4 * l_ce + 0.6 * l_dice

            # 2. Classification Loss (calculated on labeled samples only)
            valid_cls = cls_labels >= 0
            if valid_cls.any():
                l_cls = ce_loss_cls(cls_logits[valid_cls], cls_labels[valid_cls])
                preds_cls = torch.argmax(cls_logits[valid_cls], dim=1)
                train_cls_correct += (preds_cls == cls_labels[valid_cls]).sum().item()
                train_cls_count += valid_cls.sum().item()
            else:
                l_cls = torch.tensor(0.0, device=device)

            # Joint Multi-Task Loss
            loss = args.lambda_seg * l_seg + args.lambda_cls * l_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Poly LR decay
            lr_curr = args.base_lr * (1.0 - (global_step / total_steps)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_curr

            train_loss_total += loss.item()
            train_seg_total += l_seg.item()
            train_cls_total += l_cls.item()

            step_pct = (step / steps_per_epoch) * 100.0
            overall_pct = ((epoch + step / steps_per_epoch) / total_epochs) * 100.0

            # Real-time percentage terminal logging
            if step == 1 or step % 5 == 0 or step == steps_per_epoch:
                cur_acc = (train_cls_correct / train_cls_count * 100.0) if train_cls_count > 0 else 0.0
                print(
                    f"[{overall_pct:5.1f}% TOTAL] [Epoch {epoch+1:03d}/{total_epochs:03d} | Step {step:02d}/{steps_per_epoch:02d} ({step_pct:5.1f}%)] "
                    f"Loss: {loss.item():.4f} (Seg: {l_seg.item():.4f}, Cls: {l_cls.item():.4f}) | Cls Acc: {cur_acc:5.1f}% | LR: {lr_curr:.5f}",
                    flush=True,
                )

        train_epoch_loss = train_loss_total / steps_per_epoch
        train_epoch_seg = train_seg_total / steps_per_epoch
        train_epoch_cls = train_cls_total / steps_per_epoch
        train_epoch_acc = (train_cls_correct / train_cls_count * 100.0) if train_cls_count > 0 else 0.0

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_seg_total = 0.0
        val_cls_total = 0.0
        val_dice_scores = []
        val_cls_correct = 0
        val_cls_count = 0

        with torch.no_grad():
            for val_imgs, val_masks, val_cls in val_loader:
                val_imgs = val_imgs.to(device, non_blocking=True)
                val_masks = val_masks.to(device, non_blocking=True)
                val_cls = val_cls.to(device, non_blocking=True)

                v_seg_logits, v_cls_logits = model(val_imgs)

                v_ce = ce_loss_seg(v_seg_logits, val_masks)
                v_dice = dice_loss_seg(v_seg_logits, val_masks, softmax=True)
                v_seg = 0.4 * v_ce + 0.6 * v_dice

                v_valid_cls = val_cls >= 0
                if v_valid_cls.any():
                    v_cls = ce_loss_cls(v_cls_logits[v_valid_cls], val_cls[v_valid_cls])
                    v_preds = torch.argmax(v_cls_logits[v_valid_cls], dim=1)
                    val_cls_correct += (v_preds == val_cls[v_valid_cls]).sum().item()
                    val_cls_count += v_valid_cls.sum().item()
                else:
                    v_cls = torch.tensor(0.0, device=device)

                v_loss = args.lambda_seg * v_seg + args.lambda_cls * v_cls

                val_loss_total += v_loss.item()
                val_seg_total += v_seg.item()
                val_cls_total += v_cls.item()

                batch_dice = compute_dice_score(v_seg_logits, val_masks)
                val_dice_scores.append(batch_dice)

        val_epoch_loss = val_loss_total / len(val_loader)
        val_epoch_seg = val_seg_total / len(val_loader)
        val_epoch_cls = val_cls_total / len(val_loader)
        val_epoch_dice = float(np.mean(val_dice_scores)) * 100.0
        val_epoch_acc = (val_cls_correct / val_cls_count * 100.0) if val_cls_count > 0 else 0.0

        epoch_time = time.time() - epoch_start_time
        elapsed_total = time.time() - start_train_time
        completed_epochs = epoch + 1
        eta_seconds = (elapsed_total / completed_epochs) * (total_epochs - completed_epochs)
        epoch_overall_pct = (completed_epochs / total_epochs) * 100.0

        is_best = val_epoch_loss < best_val_loss
        ckpt_msg = ""
        if is_best:
            best_val_loss = val_epoch_loss
            best_val_dice = val_epoch_dice
            best_val_acc = val_epoch_acc
            best_ckpt = out_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": completed_epochs,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_epoch_loss,
                    "val_dice": val_epoch_dice,
                    "val_acc": val_epoch_acc,
                },
                str(best_ckpt),
            )
            ckpt_msg = f"🌟 NEW BEST! Saved to {best_ckpt.name}"
        else:
            ckpt_msg = f"Best Val Loss: {best_val_loss:.4f} (Dice: {best_val_dice:.2f}%, Acc: {best_val_acc:.2f}%)"

        latest_ckpt = out_dir / "latest_model.pth"
        torch.save(
            {
                "epoch": completed_epochs,
                "model_state_dict": model.state_dict(),
                "val_loss": val_epoch_loss,
                "val_dice": val_epoch_dice,
                "val_acc": val_epoch_acc,
            },
            str(latest_ckpt),
        )

        log_print("-" * 85)
        log_print(
            f"🎯 [HOÀN THÀNH EPOCH {completed_epochs:03d}/{total_epochs:03d} — TIẾN ĐỘ: {epoch_overall_pct:5.1f}%] (Thời gian: {format_time(epoch_time)} | Còn lại: ~{format_time(eta_seconds)})"
        )
        log_print(
            f"   Train: Total Loss: {train_epoch_loss:.4f} (Seg: {train_epoch_seg:.4f}, Cls: {train_epoch_cls:.4f}) | Cls Acc: {train_epoch_acc:5.2f}%"
        )
        log_print(
            f"   Valid: Total Loss: {val_epoch_loss:.4f} (Seg: {val_epoch_seg:.4f}, Cls: {val_epoch_cls:.4f}) | Silkworm Dice: {val_epoch_dice:5.2f}% | Cls Acc: {val_epoch_acc:5.2f}%"
        )
        log_print(f"   Checkpoint: {ckpt_msg}")
        log_print("-" * 85)

    total_time = time.time() - start_train_time
    log_print("\n" + "=" * 85)
    log_print(f"🎉 HUẤN LUYỆN MULTI-TASK HOÀN TẤT 100%! Tổng thời gian: {format_time(total_time)}")
    log_print(
        f" Best Val Loss: {best_val_loss:.4f} | Best Silkworm Dice: {best_val_dice:.2f}% | Best Cls Acc: {best_val_acc:.2f}%"
    )
    log_print(f" Checkpoint tốt nhất: {out_dir / 'best_model.pth'}")
    log_print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Swin-Unet Training")
    parser.add_argument("--data_dir", type=str, default="../data/silkworm_mixed_dataset")
    parser.add_argument("--cfg", type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--output_dir", type=str, default="./multitask_silkworm_out")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--seed", type=int, default=1234)

    # Required Swin-Unet dummy config arguments
    parser.add_argument("--dataset", type=str, default="silkworm_multitask")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O1")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--opts", default=None, nargs="+")

    args = parser.parse_args()
    train_multitask_silkworm(args)


if __name__ == "__main__":
    main()
