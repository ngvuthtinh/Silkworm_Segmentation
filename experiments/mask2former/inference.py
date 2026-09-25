"""
inference.py — Single-image or directory inference with Mask2Former.
Produces visual overlay of silkworm segmentation.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK2FORMER_ROOT = PROJECT_ROOT / "models" / "Mask2Former"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MASK2FORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(MASK2FORMER_ROOT))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

from experiments.mask2former.dataset_adapter import register_silkworm_mixed_datasets
from experiments.mask2former.config import setup_mask2former_cfg


def run_inference(image_path: str, checkpoint_path: str = "runs/mask2former/model_final.pth", output_path: str | None = None):
    register_silkworm_mixed_datasets()
    cfg = setup_mask2former_cfg()
    cfg.MODEL.WEIGHTS = checkpoint_path

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = T.Resize((224, 224))
    aug_input = T.AugInput(img_rgb)
    transform(aug_input)

    img_tensor = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1)).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        outputs = model([{"image": img_tensor, "height": 224, "width": 224}])
        sem_seg = outputs[0]["sem_seg"].argmax(dim=0).cpu().numpy()

    # Resize prediction back to original image size
    pred_mask = cv2.resize((sem_seg == 1).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create green overlay
    overlay = img_bgr.copy()
    overlay[pred_mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

    # Combine: Original | Mask | Overlay
    mask_bgr = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([img_bgr, mask_bgr, blended])

    if output_path is None:
        out_dir = Path("runs/mask2former/inferences")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"pred_{Path(image_path).name}")

    cv2.imwrite(output_path, combined)
    print(f"Inference saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask2Former Silkworm Inference")
    parser.add_argument("--image", type=str, default="data/test/image.png", help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="runs/mask2former/model_final.pth", help="Checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.output)
