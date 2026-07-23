"""Convert YOLO detection labels into binary masks and boundary maps.

This dataset stores standard YOLO bbox annotations, not instance masks.
For the simplified Deep Watershed pipeline, we approximate each box as a
filled foreground region, then derive a 2-pixel boundary map from that mask.

Output structure:

    output_root/
      train/
        masks/
        boundaries/
      valid/
        masks/
        boundaries/
      test/
        masks/
        boundaries/

The produced masks are suitable for boundary-label generation in this repo's
training pipeline, but they are not true pixel-accurate instance masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _find_split_images(split_dir: Path) -> list[Path]:
    image_dir = split_dir / "images"
    if not image_dir.exists():
        return []
    return [path for path in sorted(image_dir.iterdir()) if path.is_file() and _is_image_file(path)]


def _load_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return height, width


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []

    labels: list[tuple[int, float, float, float, float]] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO bbox label line in {label_path}: {raw_line!r}")
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])
        labels.append((class_id, x_center, y_center, box_width, box_height))
    return labels


def _box_to_pixels(
    image_width: int,
    image_height: int,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
) -> tuple[int, int, int, int]:
    x1 = int(round((x_center - box_width / 2.0) * image_width))
    y1 = int(round((y_center - box_height / 2.0) * image_height))
    x2 = int(round((x_center + box_width / 2.0) * image_width))
    y2 = int(round((y_center + box_height / 2.0) * image_height))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    return x1, y1, x2, y2


def convert_yolo_split(split_dir: Path, output_root: Path) -> None:
    image_paths = _find_split_images(split_dir)
    if not image_paths:
        return

    label_dir = split_dir / "labels"
    mask_dir = output_root / split_dir.name / "masks"
    boundary_dir = output_root / split_dir.name / "boundaries"
    mask_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        height, width = _load_image_size(image_path)
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = _read_yolo_labels(label_path)

        mask = np.zeros((height, width), dtype=np.uint8)
        for _, x_center, y_center, box_width, box_height in labels:
            x1, y1, x2, y2 = _box_to_pixels(width, height, x_center, y_center, box_width, box_height)
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1 : y2 + 1, x1 : x2 + 1] = 255

        boundary = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary, contours, -1, 255, thickness=2)

        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
        cv2.imwrite(str(boundary_dir / f"{image_path.stem}.png"), boundary)


def convert_dataset(input_root: str, output_root: str) -> None:
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)

    for split_name in SPLITS:
        split_dir = input_root_path / split_name
        if split_dir.exists():
            convert_yolo_split(split_dir, output_root_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert YOLO bbox labels into masks and boundaries")
    parser.add_argument("--input-root", required=True, help="Path to the YOLO dataset root containing train/valid/test")
    parser.add_argument("--output-root", required=True, help="Where masks/boundaries should be written")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    convert_dataset(args.input_root, args.output_root)


if __name__ == "__main__":
    main()
