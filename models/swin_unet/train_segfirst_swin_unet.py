#!/usr/bin/env python3
"""
train_segfirst_swin_unet.py — Segmentation-First Multi-Task Swin-Unet Training Pipeline
======================================================================================
Strategy: Two-Phase Training (Identical to SegFirst Multi-Task VM-UNet)
  - Phase 1: Segmentation Pre-training on silkworm masks (25 epochs)
             Focuses backbone on silkworm geometry & contour extraction.
  - Phase 2: Multi-Task Fine-Tuning with Differential Learning Rates (25 epochs)
             lr_backbone=1e-5, lr_seg=1e-4, lr_cls=1e-3, lambda_cls=0.1.
             Jointly performs silkworm segmentation + disease detection (Healthy vs Grasserie).

Progress: Real-time percentage (% completed) logged directly to terminal.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import get_config
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from utils import DiceLoss


class SegFirstSwinUnet(nn.Module):
    """
    SegFirst Swin-Unet Architecture:
      - SwinTransformerSys backbone & decoder for Segmentation
      - Bottleneck MLP Classification Head (768 -> 128 -> 2)
      - Phase 1 / Phase 2 flexible forward pass
      - Parameter groups with differential learning rates
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

    def forward(self, x: torch.Tensor, phase: int = 2) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. Encoder forward pass to bottleneck
        bottleneck, x_downsample = self.swin_unet.forward_features(x)

        # 2. Decoder segmentation
        x_up = self.swin_unet.forward_up_features(bottleneck, x_downsample)
        seg_logits = self.swin_unet.up_x4(x_up)

        if phase == 1:
            return seg_logits, None

        # 3. Classification Head from bottleneck global average pooling
        cls_feat = bottleneck.mean(dim=1)
        cls_logits = self.cls_head(cls_feat)

        return seg_logits, cls_logits

    def get_parameter_groups(self, lr_backbone: float, lr_seg: float, lr_cls: float) -> list[dict]:
        """
        Differential learning rates for Phase 2:
        - lr_backbone: Low LR (1e-5) to preserve spatial features learned in Phase 1
        - lr_seg: Moderate LR (1e-4) for decoder
        - lr_cls: Higher LR (1e-3) for fresh classification head
        """
        backbone_params = []
        seg_head_params = []
        cls_params = list(self.cls_head.parameters())

        for name, param in self.swin_unet.named_parameters():
            if any(k in name for k in ["layers_up", "concat_back_dim", "norm_up", "up", "output"]):
                seg_head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": seg_head_params, "lr": lr_seg, "name": "seg_head"},
            {"params": cls_params, "lr": lr_cls, "name": "cls_head"},
        ]

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
            full_dict = model_dict.copy()
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    full_dict[k] = v
            self.swin_unet.load_state_dict(full_dict, strict=False)
            print(" Pretrained weights loaded successfully into Swin-Unet backbone!\n")


class SilkwormSegFirstDataset(Dataset):
    """
    Multi-Task Dataset for Two-Phase Training.
    Returns (image, mask, class_label)
    class_label: 0=Healthy, 1=Grasserie, -1=Unlabeled
    """

    def __init__(self, data_root: str, split: str = "train", img_size: int = 224) -> None:
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.is_train = split == "train"

        root = Path(data_root) / split
        img_dir = root / "images"
        mask_dir = root / "masks"

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

            name_lower = img_p.name.lower()
            if "grasserie" in name_lower:
                cls_label = 1
            elif "healthy" in name_lower:
                cls_label = 0
            else:
                cls_label = -1

            self.samples.append((img_p, mask_p, cls_label))

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


def train_segfirst(args) -> None:
    gpu_id = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "segfirst_train.log"

    logger = logging.getLogger("SegFirst_SwinUnet")
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
    log_print("   🌟 SEGMENTATION-FIRST MULTI-TASK SWIN-UNET TRAINING PIPELINE")
    log_print("=" * 85)
    log_print(f" Dataset Root     : {args.data_dir}")
    log_print(f" Output Directory : {args.output_dir}")
    log_print(f" Phase 1 Epochs   : {args.phase1_epochs} (Segmentation Pre-training)")
    log_print(f" Phase 2 Epochs   : {args.phase2_epochs} (Multi-Task Fine-Tuning)")
    log_print(f" Batch Size       : {args.batch_size} | Image Size: {args.img_size}x{args.img_size}")
    log_print(f" Differential LRs : Backbone={args.lr_backbone} | Seg={args.lr_seg} | Cls={args.lr_cls}")
    log_print(f" Lambda Weights   : Seg={args.lambda_seg} | Cls={args.lambda_cls}")
    log_print(f" Device           : {device} (GPU ID: {gpu_id})")
    log_print("=" * 85)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_dataset = SilkwormSegFirstDataset(args.data_dir, split="train", img_size=args.img_size)
    val_dataset = SilkwormSegFirstDataset(args.data_dir, split="valid", img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    log_print(f" Train: {len(train_dataset)} samples ({len(train_loader)} batches/epoch)")
    log_print(f" Valid: {len(val_dataset)} samples ({len(val_loader)} batches/epoch)")

    config = get_config(args)
    model = SegFirstSwinUnet(
        config=config,
        img_size=args.img_size,
        num_seg_classes=2,
        num_cls_classes=2,
        cls_hidden=128,
        cls_dropout=0.3,
    ).to(device)

    model.load_from(config)

    ce_loss_seg = nn.CrossEntropyLoss()
    dice_loss_seg = DiceLoss(n_classes=2)
    ce_loss_cls = nn.CrossEntropyLoss()

    # =========================================================================
    # PHASE 1: SEGMENTATION PRE-TRAINING
    # =========================================================================
    log_print("\n" + "=" * 85)
    log_print("🚀 [PHASE 1/2] BẮT ĐẦU SEGMENTATION PRE-TRAINING (Học trích xuất đặc trưng hình thái)")
    log_print("=" * 85)

    opt_p1 = optim.SGD(model.swin_unet.parameters(), lr=args.lr_seg, momentum=0.9, weight_decay=1e-4)
    p1_total_steps = args.phase1_epochs * len(train_loader)
    p1_step = 0
    best_p1_dice = 0.0
    p1_best_path = ckpt_dir / "phase1_best.pth"
    t0_p1 = time.time()

    for ep in range(args.phase1_epochs):
        t_ep = time.time()
        model.train()
        p1_loss_total = 0.0

        for step, (images, masks, _) in enumerate(train_loader, 1):
            p1_step += 1
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            seg_logits, _ = model(images, phase=1)
            l_ce = ce_loss_seg(seg_logits, masks)
            l_dice = dice_loss_seg(seg_logits, masks, softmax=True)
            loss_p1 = 0.4 * l_ce + 0.6 * l_dice

            opt_p1.zero_grad()
            loss_p1.backward()
            opt_p1.step()

            p1_loss_total += loss_p1.item()

            step_pct = (step / len(train_loader)) * 100.0
            phase1_pct = ((ep + step / len(train_loader)) / args.phase1_epochs) * 100.0

            if step == 1 or step % 5 == 0 or step == len(train_loader):
                print(
                    f"[Phase 1: {phase1_pct:5.1f}%] [Epoch {ep+1:02d}/{args.phase1_epochs:02d} | Step {step:02d}/{len(train_loader):02d} ({step_pct:5.1f}%)] "
                    f"Seg Loss: {loss_p1.item():.4f} (CE: {l_ce.item():.4f}, Dice: {l_dice.item():.4f})",
                    flush=True,
                )

        train_p1_loss = p1_loss_total / len(train_loader)

        # Validation Phase 1
        model.eval()
        val_dices = []
        val_p1_loss = 0.0
        with torch.no_grad():
            for val_imgs, val_masks, _ in val_loader:
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)
                v_logits, _ = model(val_imgs, phase=1)
                v_loss = 0.4 * ce_loss_seg(v_logits, val_masks) + 0.6 * dice_loss_seg(v_logits, val_masks, softmax=True)
                val_p1_loss += v_loss.item()
                val_dices.append(compute_dice_score(v_logits, val_masks))

        val_p1_loss /= len(val_loader)
        mean_p1_dice = float(np.mean(val_dices)) * 100.0
        is_best_p1 = mean_p1_dice > best_p1_dice
        p1_msg = ""
        if is_best_p1:
            best_p1_dice = mean_p1_dice
            torch.save(model.state_dict(), str(p1_best_path))
            p1_msg = f"🌟 NEW BEST Phase 1! Saved to {p1_best_path.name}"
        else:
            p1_msg = f"Best Phase 1 Dice: {best_p1_dice:.2f}%"

        ep_time = time.time() - t_ep
        overall_p1_pct = ((ep + 1) / args.phase1_epochs) * 100.0
        log_print(
            f" [PHASE 1 Ep {ep+1:02d}/{args.phase1_epochs:02d} — {overall_p1_pct:5.1f}%] Train Loss: {train_p1_loss:.4f} | Val Loss: {val_p1_loss:.4f} | Silkworm Val Dice: {mean_p1_dice:.2f}% | {p1_msg}"
        )

    log_print(f"\n Phase 1 hoàn tất! Checkpoint tốt nhất lưu tại: {p1_best_path} (Dice={best_p1_dice:.2f}%)\n")

    # =========================================================================
    # PHASE 2: MULTI-TASK FINE-TUNING (DIFFERENTIAL LEARNING RATES)
    # =========================================================================
    log_print("=" * 85)
    log_print("🚀 [PHASE 2/2] BẮT ĐẦU MULTI-TASK FINE-TUNING (Differential LR + Joint Loss)")
    log_print("=" * 85)

    # Nạp lại checkpoint tốt nhất của Phase 1
    model.load_state_dict(torch.load(str(p1_best_path), map_location=device))
    log_print(" Đã nạp thành công weights từ Phase 1 vào mô hình!")

    param_groups = model.get_parameter_groups(
        lr_backbone=args.lr_backbone, lr_seg=args.lr_seg, lr_cls=args.lr_cls
    )
    opt_p2 = optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)

    p2_total_steps = args.phase2_epochs * len(train_loader)
    p2_step = 0
    best_p2_score = 0.0
    p2_best_path = ckpt_dir / "phase2_best.pth"

    for ep in range(args.phase2_epochs):
        t_ep = time.time()
        model.train()
        p2_loss_total = 0.0
        cls_correct = 0
        cls_count = 0

        for step, (images, masks, cls_labels) in enumerate(train_loader, 1):
            p2_step += 1
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)

            seg_logits, cls_logits = model(images, phase=2)

            # Seg Loss
            l_ce = ce_loss_seg(seg_logits, masks)
            l_dice = dice_loss_seg(seg_logits, masks, softmax=True)
            l_seg = 0.4 * l_ce + 0.6 * l_dice

            # Cls Loss
            valid_cls = cls_labels >= 0
            if valid_cls.any():
                l_cls = ce_loss_cls(cls_logits[valid_cls], cls_labels[valid_cls])
                preds = torch.argmax(cls_logits[valid_cls], dim=1)
                cls_correct += (preds == cls_labels[valid_cls]).sum().item()
                cls_count += valid_cls.sum().item()
            else:
                l_cls = torch.tensor(0.0, device=device)

            loss_p2 = args.lambda_seg * l_seg + args.lambda_cls * l_cls

            opt_p2.zero_grad()
            loss_p2.backward()
            opt_p2.step()

            p2_loss_total += loss_p2.item()

            step_pct = (step / len(train_loader)) * 100.0
            phase2_pct = ((ep + step / len(train_loader)) / args.phase2_epochs) * 100.0

            if step == 1 or step % 5 == 0 or step == len(train_loader):
                acc_now = (cls_correct / cls_count * 100.0) if cls_count > 0 else 0.0
                print(
                    f"[Phase 2: {phase2_pct:5.1f}%] [Epoch {ep+1:02d}/{args.phase2_epochs:02d} | Step {step:02d}/{len(train_loader):02d} ({step_pct:5.1f}%)] "
                    f"Loss: {loss_p2.item():.4f} (Seg: {l_seg.item():.4f}, Cls: {l_cls.item():.4f}) | Cls Acc: {acc_now:5.1f}%",
                    flush=True,
                )

        train_p2_loss = p2_loss_total / len(train_loader)
        train_p2_acc = (cls_correct / cls_count * 100.0) if cls_count > 0 else 0.0

        # Validation Phase 2
        model.eval()
        val_dices = []
        val_cls_correct = 0
        val_cls_count = 0
        val_p2_loss = 0.0

        with torch.no_grad():
            for val_imgs, val_masks, val_cls in val_loader:
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)
                val_cls = val_cls.to(device)

                v_seg, v_cls_out = model(val_imgs, phase=2)
                v_l_seg = 0.4 * ce_loss_seg(v_seg, val_masks) + 0.6 * dice_loss_seg(v_seg, val_masks, softmax=True)

                v_valid = val_cls >= 0
                if v_valid.any():
                    v_l_cls = ce_loss_cls(v_cls_out[v_valid], val_cls[v_valid])
                    v_preds = torch.argmax(v_cls_out[v_valid], dim=1)
                    val_cls_correct += (v_preds == val_cls[v_valid]).sum().item()
                    val_cls_count += v_valid.sum().item()
                else:
                    v_l_cls = torch.tensor(0.0, device=device)

                val_p2_loss += (args.lambda_seg * v_l_seg + args.lambda_cls * v_l_cls).item()
                val_dices.append(compute_dice_score(v_seg, val_masks))

        val_p2_loss /= len(val_loader)
        val_p2_dice = float(np.mean(val_dices)) * 100.0
        val_p2_acc = (val_cls_correct / val_cls_count * 100.0) if val_cls_count > 0 else 0.0

        # Combined Score (identical to segfirst_multitask_vmunet: dice + 0.5 * acc)
        combined_score = (val_p2_dice / 100.0) + 0.5 * (val_p2_acc / 100.0)
        is_best_p2 = combined_score > best_p2_score
        p2_msg = ""
        if is_best_p2:
            best_p2_score = combined_score
            torch.save(
                {
                    "epoch": ep + 1,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_p2_loss,
                    "val_dice": val_p2_dice,
                    "val_acc": val_p2_acc,
                    "combined_score": combined_score,
                },
                str(p2_best_path),
            )
            p2_msg = f"🌟 NEW BEST Phase 2! Saved to {p2_best_path.name}"
        else:
            p2_msg = f"Best Combined Score: {best_p2_score:.4f}"

        # Save latest
        torch.save(model.state_dict(), str(ckpt_dir / "latest.pth"))

        overall_p2_pct = ((ep + 1) / args.phase2_epochs) * 100.0
        log_print("-" * 85)
        log_print(
            f"🎯 [HOÀN THÀNH PHASE 2 Ep {ep+1:02d}/{args.phase2_epochs:02d} — {overall_p2_pct:5.1f}%]"
        )
        log_print(
            f"   Train: Loss: {train_p2_loss:.4f} | Cls Acc: {train_p2_acc:5.2f}%"
        )
        log_print(
            f"   Valid: Loss: {val_p2_loss:.4f} | Silkworm Dice: {val_p2_dice:5.2f}% | Cls Acc: {val_p2_acc:5.2f}% (Combined Score: {combined_score:.4f})"
        )
        log_print(f"   Checkpoint: {p2_msg}")
        log_print("-" * 85)

    log_print("\n" + "=" * 85)
    log_print("🎉 QUY TRÌNH HUẤN LUYỆN SEGFIRST MULTI-TASK SWIN-UNET HOÀN TẤT!")
    log_print(f" Phase 1 Best (Seg Pretrain) : {p1_best_path}")
    log_print(f" Phase 2 Best (Joint Multi)  : {p2_best_path} (Score: {best_p2_score:.4f})")
    log_print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="SegFirst Multi-Task Swin-Unet Training")
    parser.add_argument("--data_dir", type=str, default="../data/silkworm_mixed_dataset")
    parser.add_argument("--cfg", type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml")
    parser.add_argument("--output_dir", type=str, default="./segfirst_swin_out")
    parser.add_argument("--phase1_epochs", type=int, default=25)
    parser.add_argument("--phase2_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_seg", type=float, default=1e-4)
    parser.add_argument("--lr_cls", type=float, default=1e-3)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--seed", type=int, default=42)

    # Dummy args
    parser.add_argument("--dataset", type=str, default="silkworm_segfirst")
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
    train_segfirst(args)


if __name__ == "__main__":
    main()
