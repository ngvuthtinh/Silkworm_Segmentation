#!/usr/bin/env python3
"""
experiments/multitask_mask2former/train.py
End-to-End Multi-Task Mask2Former Training (Segmentation + Classification).

Chạy thử (smoke-test 5 batch):
  ./.venv/bin/python experiments/multitask_mask2former/train.py --smoke-test

Chạy thật:
  CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python experiments/multitask_mask2former/train.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MultitaskM2FDataset(Dataset):
    """
    Dataset cho Multi-Task Mask2Former:
        pixel_values: FloatTensor [3, H, W]
        sem_masks:    LongTensor  [H, W]  (0=bg, 1=Grasserie, 2=Healthy)
        class_labels: LongTensor  [1]     (0=Grasserie, 1=Healthy)
    """

    YOLO_TO_SEM = {0: 1, 1: 2}

    def __init__(
        self,
        data_root: Path,
        split: str,
        img_size: int = 256,
        augment: bool = True,
    ):
        import random as _random
        self._random = _random
        self.img_size = img_size
        self.augment  = augment and split == "train"

        split_dir = data_root / split
        img_dir   = split_dir / "images"
        mask_dir  = split_dir / "masks"
        lbl_dir   = split_dir / "labels"

        self.samples: list[tuple[Path, Path, int]] = []
        for img_p in sorted(img_dir.iterdir()):
            if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = img_p.stem
            mask_p = mask_dir / f"{stem}.png"
            if not mask_p.exists():
                continue
            lbl_p = lbl_dir / f"{stem}.txt"
            class_id = self._read_class(lbl_p, stem)
            self.samples.append((img_p, mask_p, class_id))

    @staticmethod
    def _read_class(lbl_p: Path, stem: str) -> int:
        if lbl_p.exists():
            try:
                with open(lbl_p) as f:
                    first = f.readline().strip()
                    if first:
                        return int(first.split()[0])
            except Exception:
                pass
        return 0 if "healthy" not in stem.lower() else 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, yolo_cls = self.samples[idx]

        img  = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        s = self.img_size
        img  = cv2.resize(img,  (s, s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if self._random.random() < 0.5:
                img  = cv2.flip(img,  1); mask = cv2.flip(mask, 1)
            if self._random.random() < 0.5:
                img  = cv2.flip(img,  0); mask = cv2.flip(mask, 0)
            k = self._random.randint(0, 3)
            if k:
                img  = np.rot90(img,  k).copy()
                mask = np.rot90(mask, k).copy()

        img_f = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
        pixel_values = torch.from_numpy(img_f.transpose(2, 0, 1))

        sem_id = self.YOLO_TO_SEM[yolo_cls]
        binary = (mask > 127).astype(np.int64)
        sem_mask = torch.from_numpy(binary * sem_id)

        class_label = torch.tensor([yolo_cls], dtype=torch.long)
        return pixel_values, sem_mask, class_label


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    sem_masks    = torch.stack([b[1] for b in batch])
    class_labels = torch.stack([b[2] for b in batch])
    return pixel_values, sem_masks, class_labels


# ─── Hàm mất mát ──────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_proba: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tgt = (target > 0).float().unsqueeze(1)
        p   = pred_proba.contiguous().view(-1)
        t   = tgt.contiguous().view(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class MultitaskM2FLoss(nn.Module):
    def __init__(self, num_classes=3, lambda_seg=1.0, lambda_cls=1.0):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.ce_seg  = nn.CrossEntropyLoss(ignore_index=255)
        self.dice    = DiceLoss()
        self.ce_cls  = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, seg_logits, sem_mask, cls_logits, cls_gt):
        l_seg_ce = self.ce_seg(seg_logits, sem_mask)
        fg_proba = torch.sigmoid(seg_logits[:, 1:2, :, :] + seg_logits[:, 2:3, :, :])
        l_dice   = self.dice(fg_proba, sem_mask)
        l_seg    = l_seg_ce + l_dice
        l_cls    = self.ce_cls(cls_logits, cls_gt.squeeze(1))
        return self.lambda_seg * l_seg + self.lambda_cls * l_cls, l_seg, l_cls


# ─── Model Mask2Former-style Architecture ─────────────────────────────────────

class MultitaskM2FModel(nn.Module):
    def __init__(self, num_seg_classes: int = 3, num_cls_classes: int = 2):
        super().__init__()
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=False)
        # Load local cached weights if available
        cache_pths = [
            Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth",
            Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
        ]
        for cp in cache_pths:
            if cp.exists():
                try:
                    state = torch.load(cp, map_location="cpu")
                    backbone.load_state_dict(state, strict=False)
                    break
                except Exception:
                    pass

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = self._up_block(2048, 512)
        self.up3 = self._up_block(512 + 1024, 256)
        self.up2 = self._up_block(256 + 512, 128)
        self.up1 = self._up_block(128 + 256, 64)
        self.seg_head = nn.Conv2d(64, num_seg_classes, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_cls_classes),
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        cls_logits = self.cls_head(x4)

        d = F.interpolate(self.up4(x4), size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d = F.interpolate(self.up3(torch.cat([d, x3], 1)), size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d = F.interpolate(self.up2(torch.cat([d, x2], 1)), size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d = F.interpolate(self.up1(torch.cat([d, x1], 1)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        seg_logits = self.seg_head(d)

        return seg_logits, cls_logits


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    DATA_DIR = ROOT / "data" / "Silkworm_mixed_dataset_10k"
    IMG_SIZE = 256
    BATCH    = 8 if not args.smoke_test else 2
    EPOCHS   = 40 if not args.smoke_test else 1
    LR       = 1e-4
    WORKERS  = 4
    SMOKE_N  = 5

    device = _pick_device(args)
    print(f"\n{'='*70}")
    print(f" 🐛 Multi-Task Mask2Former — End-to-End Training")
    print(f"    Dataset : {DATA_DIR}")
    print(f"    Img Size: {IMG_SIZE}×{IMG_SIZE}  |  Batch: {BATCH}  |  Epochs: {EPOCHS}")
    print(f"    Device  : {device}")
    print(f"    Mode    : {'SMOKE-TEST (5 batch)' if args.smoke_test else 'FULL TRAIN'}")
    print(f"{'='*70}\n")

    ds_train = MultitaskM2FDataset(DATA_DIR, "train", IMG_SIZE, augment=True)
    ds_val   = MultitaskM2FDataset(DATA_DIR, "valid", IMG_SIZE, augment=False)
    dl_train = DataLoader(ds_train, BATCH, shuffle=True,  num_workers=WORKERS, pin_memory=True, collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val,   BATCH, shuffle=False, num_workers=WORKERS, pin_memory=True, collate_fn=collate_fn)
    print(f"  Train: {len(ds_train)} mẫu ({len(dl_train)} batches)  |  Val: {len(ds_val)} mẫu")

    model = MultitaskM2FModel(num_seg_classes=3, num_cls_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params : {total_params:.1f}M  (ResNet-50 backbone + FPN decoder + cls head)\n")

    loss_fn   = MultitaskM2FLoss(num_classes=3, lambda_seg=1.0, lambda_cls=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = amp.GradScaler()

    run_tag  = "smoke_test" if args.smoke_test else datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir  = ROOT / "runs" / "multitask_mask2former" / run_tag
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        n_batches = min(SMOKE_N, len(dl_train)) if args.smoke_test else len(dl_train)
        pct_epoch = epoch / EPOCHS * 100

        pbar = tqdm(
            enumerate(dl_train),
            total=n_batches,
            desc=f"[Mask2Former] Epoch {epoch:02d}/{EPOCHS} ({pct_epoch:5.1f}%) [TRAIN]",
            ncols=110,
            leave=True,
        )
        for i, (pixel_vals, sem_masks, class_labels) in pbar:
            if i >= n_batches:
                break
            pixel_vals   = pixel_vals.to(device)
            sem_masks    = sem_masks.to(device)
            class_labels = class_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=device.type == "cuda"):
                seg_logits, cls_logits = model(pixel_vals)
                loss, l_seg, l_cls = loss_fn(seg_logits, sem_masks, cls_logits, class_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "seg":  f"{l_seg.item():.3f}",
                "cls":  f"{l_cls.item():.3f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
            })

        scheduler.step()
        mean_loss = t_loss / n_batches

        if args.smoke_test:
            print(f"  [SMOKE] Epoch {epoch} OK — loss={mean_loss:.4f}")
            break

        model.eval()
        val_iou = val_acc = 0.0
        v_steps = 0
        with torch.no_grad():
            for pixel_vals, sem_masks, class_labels in tqdm(dl_val, desc="  → [Mask2Former] Val ", ncols=110, leave=False):
                pixel_vals   = pixel_vals.to(device)
                sem_masks    = sem_masks.to(device)
                class_labels = class_labels.to(device)
                with amp.autocast(enabled=device.type == "cuda"):
                    seg_logits, cls_logits = model(pixel_vals)

                pred_cls = seg_logits.argmax(1)
                fg_pred  = (pred_cls > 0).float()
                fg_gt    = (sem_masks > 0).float()
                inter    = (fg_pred * fg_gt).sum()
                union    = (fg_pred + fg_gt).clamp(0, 1).sum()
                val_iou += (inter / (union + 1e-6)).item()

                val_acc += (cls_logits.argmax(1) == class_labels.squeeze(1)).float().mean().item()
                v_steps += 1

        val_iou /= v_steps
        val_acc /= v_steps
        combined = 0.7 * val_iou + 0.3 * val_acc

        best_tag = ""
        if combined > best_score:
            best_score = combined
            torch.save({"epoch": epoch, "model": model.state_dict(), "score": best_score},
                       ckpt_dir / "best.pth")
            best_tag = " ★ BEST"

        print(
            f"  [Mask2Former] Ep {epoch:02d}/{EPOCHS} "
            f"({pct_epoch:5.1f}%) | "
            f"Loss={mean_loss:.4f} | FgIoU={val_iou:.4f} | Acc={val_acc:.4f} | "
            f"Score={combined:.4f}{best_tag}"
        )

        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       ckpt_dir / f"ep{epoch:03d}.pth")

    torch.save({"epoch": EPOCHS, "model": model.state_dict()}, ckpt_dir / "last.pth")
    if not args.smoke_test:
        print(f"\n✅ [Mask2Former] Hoàn thành! Best score={best_score:.4f} | Saved: {out_dir}")
    else:
        print(f"\n✅ [Mask2Former] Smoke-test PASSED!")


def _pick_device(args):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if hasattr(args, "gpu") and str(args.gpu).isdigit():
        idx = int(args.gpu)
        if idx < torch.cuda.device_count():
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}")
    best, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best = i
    torch.cuda.set_device(best)
    return torch.device(f"cuda:{best}")


def main():
    p = argparse.ArgumentParser(description="Multi-Task Mask2Former Trainer")
    p.add_argument("--smoke-test", action="store_true", help="Chạy thử 5 batch")
    p.add_argument("--gpu", type=str, default="auto")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
