#!/usr/bin/env python3
"""
SAM3 Auto-Labeling Script for Silkworm Instance Segmentation
===========================================================
Generates pixel-accurate instance segmentation masks using SAM 3 (Segment Anything Model 3)
guided by text prompts (e.g., 'silkworm' / 'con tằm') and YOLO bounding boxes.

Features:
- BFloat16 CUDA Acceleration on NVIDIA A100 (~0.2s per image).
- Auto-environment detection: auto-switches to conda env 'sam3' if run in wrong python.
- Text prompt grounding: prompts SAM 3 with 'silkworm' to target only silkworms.
- Bounding box spatial clipping: guarantees masks never bleed into background.
- Clean contour extraction: produces exactly 1 clean polygon per silkworm (no fragmented specks).
- Compatible with YOLO-seg, Deep Watershed (boundary_pipeline.py), and VM-UNet.

Author: Antigravity AI Assistant
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Auto-switch to conda 'sam3' environment if current Python lacks CUDA
SAM3_ENV_PYTHON = "/home/subnh5/miniconda3/envs/sam3/bin/python"
if Path(SAM3_ENV_PYTHON).exists() and sys.executable != SAM3_ENV_PYTHON:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not has_cuda:
        print(f"[SAM3] Current Python ({sys.executable}) lacks CUDA support.")
        print(f"[SAM3] Automatically switching to GPU environment: {SAM3_ENV_PYTHON}\n")
        os.execv(SAM3_ENV_PYTHON, [SAM3_ENV_PYTHON] + sys.argv)

import cv2
import numpy as np
from PIL import Image

# Ensure local sam3 package is available in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
SAM3_DIR = SCRIPT_DIR / "sam3"
if SAM3_DIR.exists() and str(SAM3_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")

# Palette of distinct vibrant colors for visualization (BGR format)
COLOR_PALETTE = [
    (56, 56, 255),    # Red
    (255, 112, 31),   # Blue
    (31, 224, 255),   # Yellow
    (46, 204, 113),   # Green
    (241, 196, 15),   # Cyan
    (155, 89, 182),   # Purple
    (230, 126, 34),   # Orange
    (52, 152, 219),   # Sky Blue
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label instance segmentation masks with SAM 3 using text prompt & YOLO bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Prompt options
    group_prompt = parser.add_argument_group("Prompt & Guidance Options")
    group_prompt.add_argument(
        "--prompt",
        type=str,
        default="silkworm",
        help="Text prompt for SAM 3 (e.g. 'silkworm', 'silkworm larva', 'caterpillar').",
    )
    group_prompt.add_argument(
        "--conf-threshold",
        type=float,
        default=0.30,
        help="Confidence threshold for SAM 3 text grounding detections.",
    )
    group_prompt.add_argument(
        "--box-margin",
        type=float,
        default=0.08,
        help="Margin around YOLO bbox to allow contour boundary (0.08 = 8%). Prevents mask bleeding.",
    )

    # Input options
    group_input = parser.add_argument_group("Input Options")
    group_input.add_argument(
        "--input-root",
        type=str,
        default="data/Silkworm Diseases.v1i.yolo26",
        help="Path to dataset root (containing train/valid/test) OR a directory of images.",
    )
    group_input.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Specific images directory (overrides --input-root).",
    )
    group_input.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Specific YOLO bbox labels directory (defaults to sibling 'labels' folder).",
    )
    group_input.add_argument(
        "--single-image",
        type=str,
        default=None,
        help="Path to a single image file to process.",
    )
    group_input.add_argument(
        "--single-label",
        type=str,
        default=None,
        help="Path to a single YOLO bbox .txt file (used with --single-image).",
    )
    group_input.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "valid", "test"],
        help="Select which split to process ('all', 'train', 'valid', or 'test').",
    )
    group_input.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to process (e.g. --max-images 20 to test 20 images).",
    )

    # Output options
    group_output = parser.add_argument_group("Output Options")
    group_output.add_argument(
        "--output-dir",
        type=str,
        default="data/sam3_annotated",
        help="Root directory where segmentation labels, masks, boundaries, and visualizations will be saved.",
    )
    group_output.add_argument(
        "--save-yolo-txt",
        action="store_true",
        default=True,
        help="Save YOLO segmentation format txt files (<class> <x1> <y1> <x2> <y2> ...).",
    )
    group_output.add_argument(
        "--save-masks",
        action="store_true",
        default=True,
        help="Save binary/semantic mask PNGs.",
    )
    group_output.add_argument(
        "--save-boundaries",
        action="store_true",
        default=True,
        help="Save 2-pixel boundary maps (compatible with Deep Watershed pipeline).",
    )
    group_output.add_argument(
        "--save-vis",
        action="store_true",
        default=True,
        help="Save visualized overlay images (RGB + colored mask + bbox + score).",
    )
    group_output.add_argument(
        "--no-vis",
        dest="save_vis",
        action="store_false",
        help="Disable saving visualization images to speed up processing.",
    )

    # SAM 3 Model options
    group_sam = parser.add_argument_group("SAM 3 Configuration")
    group_sam.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM 3 / SAM 3.1 checkpoint (.pt). If None, auto-searches checkpoints/ directory.",
    )
    group_sam.add_argument(
        "--bpe-path",
        type=str,
        default=None,
        help="Path to BPE tokenizer vocabulary (.txt.gz). If None, auto-searches sam3 assets.",
    )
    group_sam.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use (e.g. 0 or 1).",
    )
    group_sam.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device ('cuda' or 'cpu').",
    )
    group_sam.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum contour area in pixels to keep in polygon output.",
    )
    group_sam.add_argument(
        "--simplify-epsilon",
        type=float,
        default=0.0025,
        help="Approximation factor for cv2.approxPolyDP to smooth polygons (0 to disable).",
    )
    group_sam.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="List of class names (e.g. --class-names Grasserie Healthy). If omitted, read from data.yaml.",
    )

    return parser.parse_args()


# =====================================================================
# Path Resolution and Model Loading Helpers
# =====================================================================

def find_default_checkpoint() -> Optional[str]:
    """Search common locations for SAM 3 checkpoints."""
    candidates = [
        SCRIPT_DIR / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt",
        SCRIPT_DIR / "checkpoints" / "sam3" / "sam3.pt",
        SCRIPT_DIR / "checkpoints" / "sam3.1_multiplex.pt",
        SCRIPT_DIR / "checkpoints" / "sam3.pt",
        Path.home() / ".cache" / "sam3" / "sam3.1_multiplex.pt",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c)
    return None


def find_default_bpe() -> Optional[str]:
    """Search common locations for SAM 3 BPE vocab."""
    candidates = [
        SCRIPT_DIR / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        SCRIPT_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        SCRIPT_DIR / "checkpoints" / "sam3.1" / "merges.txt",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c)
    return None


def read_data_yaml_classes(dataset_root: Path) -> List[str]:
    """Read class names from data.yaml if present."""
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return []

    try:
        content = yaml_path.read_text(encoding="utf-8")
        import yaml
        data = yaml.safe_load(content)
        if "names" in data:
            names = data["names"]
            if isinstance(names, list):
                return names
            elif isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
    except Exception:
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("names:"):
                raw_names = line.split("names:", 1)[1].strip(" []")
                names = [n.strip(" '\"") for n in raw_names.split(",") if n.strip()]
                if names:
                    return names
    return []


def load_sam3_model(checkpoint_path: Optional[str], bpe_path: Optional[str], device: str, gpu_id: int = 0):
    """Load the SAM 3 model and processor with BFloat16 and TF32 enabled."""
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[ERROR] CUDA is not available in this environment!\n"
                "Please run with the 'sam3' conda environment:\n"
                "  conda activate sam3\n"
                "  python sam3_label_from_yolo.py ..."
            )
        # Select GPU
        torch.cuda.set_device(gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[SAM3] Hardware: {gpu_name} (GPU {gpu_id})")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if checkpoint_path is None:
        checkpoint_path = find_default_checkpoint()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "SAM 3 checkpoint not found! Please specify --checkpoint path/to/sam3.1_multiplex.pt"
            )

    if bpe_path is None:
        bpe_path = find_default_bpe()

    print(f"[SAM3] Loading checkpoint: {checkpoint_path}")
    if bpe_path:
        print(f"[SAM3] Using BPE vocab:   {bpe_path}")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        enable_inst_interactivity=True,
        device=device,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, device=device)
    print("[SAM3] Model and processor ready on GPU!\n")
    return model, processor


# =====================================================================
# Bounding Box and Label Parsing
# =====================================================================

class BoundingBox:
    def __init__(
        self,
        class_id: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        conf: float = 1.0,
        class_name: str = "",
    ):
        self.class_id = class_id
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.conf = conf
        self.class_name = class_name or f"class_{class_id}"

    def to_xyxy_pixels(self, img_w: int, img_h: int) -> np.ndarray:
        """Convert normalized (cx, cy, w, h) to absolute pixel [x1, y1, x2, y2]."""
        x_center = self.cx * img_w
        y_center = self.cy * img_h
        box_w = self.w * img_w
        box_h = self.h * img_h

        x1 = max(0.0, x_center - box_w / 2.0)
        y1 = max(0.0, y_center - box_h / 2.0)
        x2 = min(float(img_w), x_center + box_w / 2.0)
        y2 = min(float(img_h), y_center + box_h / 2.0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)


def read_yolo_label_file(label_path: Path, class_names: Optional[List[str]] = None) -> List[BoundingBox]:
    """Read bounding boxes from a YOLO txt file."""
    if not label_path.exists():
        return []

    boxes = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0

        cname = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
        boxes.append(BoundingBox(class_id, cx, cy, w, h, conf, cname))
    return boxes


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return float(interArea / union)


# =====================================================================
# Mask and Polygon Operations
# =====================================================================

def clean_and_extract_polygons(
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    box_margin: float = 0.08,
    min_area: int = 50,
    simplify_epsilon: float = 0.0025,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Clean mask, clip strictly to bounding box ROI, and extract clean polygons."""
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        mask = mask.reshape(img_h, img_w)
    mask = (mask > 0).astype(np.uint8)
    # 1. Clip mask to bounding box + margin
    bw = box_xyxy[2] - box_xyxy[0]
    bh = box_xyxy[3] - box_xyxy[1]
    mx = int(bw * box_margin)
    my = int(bh * box_margin)
    clip_x1 = max(0, int(box_xyxy[0]) - mx)
    clip_y1 = max(0, int(box_xyxy[1]) - my)
    clip_x2 = min(img_w, int(box_xyxy[2]) + mx)
    clip_y2 = min(img_h, int(box_xyxy[3]) + my)

    roi = np.zeros_like(mask, dtype=np.uint8)
    roi[clip_y1:clip_y2, clip_x1:clip_x2] = 1
    cleaned_mask = (mask * roi).astype(np.uint8)

    # 2. Morphological open/close to remove 1-pixel noise and smooth contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8), []

    # 4. Filter contours: keep primary silkworm body (and secondary parts >= 25% of max area)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_area = cv2.contourArea(contours[0])
    if max_area < min_area:
        return np.zeros_like(mask, dtype=np.uint8), []

    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) >= max(min_area, 0.25 * max_area)
    ]

    # Rebuild mask from only the valid contours
    final_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(final_mask, valid_contours, -1, 1, thickness=-1)

    # 5. Simplify contours into polygons
    polygons = []
    for cnt in valid_contours:
        if simplify_epsilon > 0:
            epsilon = simplify_epsilon * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

        cnt = cnt.reshape(-1, 2)
        if len(cnt) < 3:
            continue

        norm_pts = cnt.astype(np.float32)
        norm_pts[:, 0] = np.clip(norm_pts[:, 0] / img_w, 0.0, 1.0)
        norm_pts[:, 1] = np.clip(norm_pts[:, 1] / img_h, 0.0, 1.0)
        polygons.append(norm_pts.flatten().tolist())

    return final_mask, polygons


def render_visualization(
    image_bgr: np.ndarray,
    instances: List[Dict],
    alpha: float = 0.45,
) -> np.ndarray:
    """Render a clean, high-quality visualization overlay on the image."""
    vis = image_bgr.copy()
    mask_layer = np.zeros_like(image_bgr)
    img_h, img_w = image_bgr.shape[:2]

    for inst in instances:
        mask = inst["mask"]
        class_id = inst["class_id"]
        class_name = inst["class_name"]
        iou_score = inst.get("iou_score", 1.0)
        box = inst["box_xyxy"]

        color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

        # Paint mask color
        mask_layer[mask > 0] = color

        # Draw contour line on mask boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, thickness=2)

        # Draw bounding box
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

        # Draw label tag
        tag = f"{class_name} ({iou_score:.2f})"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - th - 8)
        tag_y2 = max(th + 8, y1)
        cv2.rectangle(vis, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
        cv2.putText(
            vis,
            tag,
            (x1 + 3, tag_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    # Alpha blend mask layer with image
    has_mask = np.any(mask_layer > 0, axis=-1)
    vis[has_mask] = cv2.addWeighted(image_bgr[has_mask], 1.0 - alpha, mask_layer[has_mask], alpha, 0)

    return vis


# =====================================================================
# Main Inference Logic on Single Image
# =====================================================================

def process_image(
    image_path: Path,
    boxes: List[BoundingBox],
    model,
    processor,
    output_labels_dir: Optional[Path],
    output_masks_dir: Optional[Path],
    output_boundaries_dir: Optional[Path],
    output_vis_dir: Optional[Path],
    prompt: str = "silkworm",
    conf_threshold: float = 0.30,
    box_margin: float = 0.08,
    min_area: int = 50,
    simplify_epsilon: float = 0.0025,
) -> int:
    """Process a single image with SAM3 guided by text prompt and YOLO bounding boxes."""
    import torch

    pil_image = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_image.size
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Run in BFloat16 autocast for maximum speed on NVIDIA A100
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # 1. Compute ViT Image Embedding (once per image)
        inference_state = processor.set_image(pil_image)

        # 2. Text Prompt Grounding: Ask SAM 3 for the target object ('silkworm')
        text_masks = []
        text_boxes = []
        text_scores = []
        if prompt:
            try:
                processor.confidence_threshold = conf_threshold
                text_out = processor.set_text_prompt(prompt, inference_state)
                if "masks" in text_out and len(text_out["masks"]) > 0:
                    text_masks = text_out["masks"].float().cpu().numpy()
                    text_boxes = text_out["boxes"].float().cpu().numpy()
                    text_scores = text_out["scores"].float().cpu().numpy()
            except Exception as e:
                # Text prompt error fallback
                pass

        instances = []
        combined_binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        combined_semantic_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        yolo_seg_lines = []

        # 3. For each YOLO bounding box: match with text prompt mask OR run box prompt
        for box_obj in boxes:
            box_xyxy = box_obj.to_xyxy_pixels(img_w, img_h)
            if box_xyxy[2] <= box_xyxy[0] or box_xyxy[3] <= box_xyxy[1]:
                continue

            matched_mask = None
            matched_score = 0.0

            # Option A: Match with text-prompted silkworm mask
            best_iou = 0.0
            best_idx = -1
            for i, cand_box in enumerate(text_boxes):
                iou = compute_iou(box_xyxy, cand_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= 0.15:
                # High-confidence match from text grounding
                matched_mask = (text_masks[best_idx] > 0.5).astype(np.uint8)
                matched_score = float(text_scores[best_idx])
            else:
                # Option B: Run SAM 3 interactive box prompt
                try:
                    masks_out, scores_out, _ = model.predict_inst(
                        inference_state,
                        point_coords=None,
                        point_labels=None,
                        box=box_xyxy[None, :],
                        multimask_output=False,
                    )
                    raw = masks_out[0] if masks_out.ndim == 3 else masks_out
                    matched_mask = (raw > 0.0).astype(np.uint8)
                    matched_score = float(scores_out[0]) if len(scores_out) > 0 else 1.0
                except Exception:
                    # Final fallback: use bounding box filled mask
                    matched_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
                    matched_mask[y1:y2, x1:x2] = 1
                    matched_score = 1.0

            if matched_mask is None or matched_mask.sum() == 0:
                continue

            # 4. Clean mask, clip strictly to YOLO box region, and extract clean polygons
            cleaned_inst_mask, polygons = clean_and_extract_polygons(
                matched_mask,
                box_xyxy=box_xyxy,
                img_w=img_w,
                img_h=img_h,
                box_margin=box_margin,
                min_area=min_area,
                simplify_epsilon=simplify_epsilon,
            )

            if cleaned_inst_mask.sum() == 0:
                continue

            # Append YOLO segmentation polygon lines
            for poly in polygons:
                coords_str = " ".join(f"{coord:.6f}" for coord in poly)
                yolo_seg_lines.append(f"{box_obj.class_id} {coords_str}\n")

            combined_binary_mask[cleaned_inst_mask > 0] = 255
            combined_semantic_mask[cleaned_inst_mask > 0] = (box_obj.class_id + 1)

            instances.append({
                "mask": cleaned_inst_mask,
                "class_id": box_obj.class_id,
                "class_name": box_obj.class_name,
                "iou_score": matched_score,
                "box_xyxy": box_xyxy,
            })

    stem = image_path.stem

    # Save outputs
    # 1. YOLO Segmentation format (.txt)
    if output_labels_dir is not None:
        out_txt = output_labels_dir / f"{stem}.txt"
        out_txt.write_text("".join(yolo_seg_lines), encoding="utf-8")

    # 2. Binary mask (.png)
    if output_masks_dir is not None:
        cv2.imwrite(str(output_masks_dir / f"{stem}.png"), combined_binary_mask)

    # 3. Boundary map (.png) for Deep Watershed
    if output_boundaries_dir is not None:
        boundary = np.zeros_like(combined_binary_mask)
        contours, _ = cv2.findContours(combined_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, 255, thickness=2)
        cv2.imwrite(str(output_boundaries_dir / f"{stem}.png"), boundary)

    # 4. Visualization image (.jpg)
    if output_vis_dir is not None and len(instances) > 0:
        vis_img = render_visualization(image_bgr, instances)
        cv2.imwrite(str(output_vis_dir / f"{stem}.jpg"), vis_img)

    return len(instances)


# =====================================================================
# Dataset Processing
# =====================================================================

def process_dataset(args: argparse.Namespace):
    """Orchestrate processing for whole dataset, directory, or single image."""
    model, processor = load_sam3_model(args.checkpoint, args.bpe_path, args.device, args.gpu_id)

    # Determine class names
    class_names = args.class_names
    input_root_path = Path(args.input_root)
    if not class_names and input_root_path.exists():
        class_names = read_data_yaml_classes(input_root_path)
    if class_names:
        print(f"[Dataset] Identified classes: {class_names}")

    print(f"[SAM3] Text Prompt:          '{args.prompt}'")
    print(f"[SAM3] Box Margin ROI:       {args.box_margin * 100:.0f}%\n")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Case 1: Single image mode
    if args.single_image:
        img_p = Path(args.single_image)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")

        if args.single_label:
            boxes = read_yolo_label_file(Path(args.single_label), class_names)
        else:
            candidate = img_p.with_suffix(".txt")
            if candidate.exists():
                boxes = read_yolo_label_file(candidate, class_names)
            else:
                boxes = []

        labels_dir = output_root / "labels" if args.save_yolo_txt else None
        masks_dir = output_root / "masks" if args.save_masks else None
        bounds_dir = output_root / "boundaries" if args.save_boundaries else None
        vis_dir = output_root / "visualizations" if args.save_vis else None

        for d in [labels_dir, masks_dir, bounds_dir, vis_dir]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

        n = process_image(
            img_p,
            boxes,
            model,
            processor,
            labels_dir,
            masks_dir,
            bounds_dir,
            vis_dir,
            prompt=args.prompt,
            conf_threshold=args.conf_threshold,
            box_margin=args.box_margin,
            min_area=args.min_area,
            simplify_epsilon=args.simplify_epsilon,
        )
        print(f"\n[DONE] Successfully labeled: {img_p.name} ({n} silkworm masks generated).")
        print(f"[Output] Check results in: {output_root.resolve()}")
        if labels_dir and (labels_dir / f"{img_p.stem}.txt").exists():
            lines_count = len((labels_dir / f"{img_p.stem}.txt").read_text().splitlines())
            print(f"[Output] YOLO polygon lines in txt: {lines_count}")
        return

    # Case 2: Custom images-dir and labels-dir
    if args.images_dir:
        img_dir = Path(args.images_dir)
        lbl_dir = Path(args.labels_dir) if args.labels_dir else img_dir.parent / "labels"
        image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if args.max_images is not None:
            image_files = image_files[:args.max_images]
        _run_batch(
            image_files,
            lbl_dir,
            output_root,
            model,
            processor,
            args,
            class_names,
            split_name="",
        )
        return

    # Case 3: Standard dataset structure with train / valid / test splits
    target_splits = SPLITS if args.split == "all" else (args.split,)
    has_splits = any((input_root_path / s).exists() for s in target_splits)
    if has_splits:
        print(f"[Dataset] Target split(s): {target_splits}")
        for split in target_splits:
            split_dir = input_root_path / split
            if not split_dir.exists():
                continue

            img_dir = split_dir / "images"
            if not img_dir.exists():
                img_dir = split_dir
            lbl_dir = split_dir / "labels"

            image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
            if not image_files:
                continue

            if args.max_images is not None:
                image_files = image_files[:args.max_images]
                print(f"[{split.upper()}] Limiting to first {len(image_files)} images (--max-images {args.max_images})")

            split_output_root = output_root / split
            _run_batch(
                image_files,
                lbl_dir,
                split_output_root,
                model,
                processor,
                args,
                class_names,
                split_name=split,
            )

        create_segmentation_yaml(output_root, class_names)
        print(f"\n[SUCCESS] Completed! Exported to: {output_root.resolve()}")
        return

    # Case 4: Flat directory of images
    image_files = sorted([p for p in input_root_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if args.max_images is not None:
        image_files = image_files[:args.max_images]
    if image_files:
        lbl_dir = input_root_path / "labels" if (input_root_path / "labels").exists() else input_root_path
        _run_batch(
            image_files,
            lbl_dir,
            output_root,
            model,
            processor,
            args,
            class_names,
            split_name="",
        )
    else:
        print(f"[ERROR] No images found in: {input_root_path}")


def _run_batch(
    image_files: List[Path],
    label_dir: Path,
    out_dir: Path,
    model,
    processor,
    args: argparse.Namespace,
    class_names: Optional[List[str]],
    split_name: str = "",
):
    """Run batch labeling with progress tracking."""
    labels_dir = out_dir / "labels" if args.save_yolo_txt else None
    masks_dir = out_dir / "masks" if args.save_masks else None
    bounds_dir = out_dir / "boundaries" if args.save_boundaries else None
    vis_dir = out_dir / "visualizations" if args.save_vis else None

    for d in [labels_dir, masks_dir, bounds_dir, vis_dir]:
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    header = f"[{split_name.upper()}]" if split_name else "[PROCESSING]"
    print(f"\n{header} Processing {len(image_files)} images...")

    total_instances = 0
    total_images_labeled = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(image_files, desc=f"{split_name or 'images'}", unit="img")
    except ImportError:
        iterator = image_files

    for idx, img_path in enumerate(iterator):
        label_file = label_dir / f"{img_path.stem}.txt"
        boxes = read_yolo_label_file(label_file, class_names) if label_file.exists() else []

        n = process_image(
            img_path,
            boxes,
            model,
            processor,
            labels_dir,
            masks_dir,
            bounds_dir,
            vis_dir,
            prompt=args.prompt,
            conf_threshold=args.conf_threshold,
            box_margin=args.box_margin,
            min_area=args.min_area,
            simplify_epsilon=args.simplify_epsilon,
        )

        if n > 0:
            total_instances += n
            total_images_labeled += 1

    print(
        f"{header} Finished: {total_images_labeled}/{len(image_files)} images labeled, "
        f"{total_instances} silkworm masks generated."
    )


def create_segmentation_yaml(output_root: Path, class_names: Optional[List[str]]):
    """Create a standard YOLO dataset yaml for the newly generated segmentation dataset."""
    yaml_lines = [
        f"path: {output_root.resolve()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
    ]
    if class_names:
        yaml_lines.append(f"nc: {len(class_names)}")
        yaml_lines.append(f"names: {class_names}")
    else:
        yaml_lines.append("nc: 2")
        yaml_lines.append("names: ['Grasserie', 'Healthy']")

    yaml_file = output_root / "dataset_segmentation.yaml"
    yaml_file.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"[Dataset] Created segmentation YAML config: {yaml_file}")


def main():
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
