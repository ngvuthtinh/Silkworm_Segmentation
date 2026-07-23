"""Silkworm boundary training and simplified Deep Watershed pipeline.

This module provides three connected pieces:
1. boundary-label generation from binary object masks,
2. a PyTorch dataset and training loop for AttentionUNet,
3. inference and post-processing with watershed separation and lesion scan.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import AttentionUNet, DiceLoss

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _read_grayscale(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return mask


def _resize(image: np.ndarray, size: Optional[tuple[int, int]]) -> np.ndarray:
    if size is None:
        return image
    height, width = size
    interpolation = cv2.INTER_AREA if image.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(image, (width, height), interpolation=interpolation)


def generate_boundary_labels(mask_dir: str, output_dir: str) -> None:
    """Create sharp boundary maps from standard binary masks.

    Each object mask is converted into a thin contour map on black background
    using cv2.findContours and a thickness of 2 pixels.
    """
    mask_dir_path = Path(mask_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for mask_path in sorted(mask_dir_path.iterdir()):
        if not mask_path.is_file() or not _is_image_file(mask_path):
            continue

        mask = _read_grayscale(mask_path)
        _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(binary_mask, dtype=np.uint8)
        cv2.drawContours(boundary, contours, -1, 255, thickness=2)

        output_path = output_dir_path / mask_path.name
        cv2.imwrite(str(output_path), boundary)


class SilkwormBoundaryDataset(Dataset):
    """Dataset returning RGB images and boundary ground-truth tensors."""

    def __init__(
        self,
        image_dir: str,
        boundary_dir: str,
        image_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.boundary_dir = Path(boundary_dir)
        self.image_size = image_size

        self.image_paths = [path for path in sorted(self.image_dir.iterdir()) if path.is_file() and _is_image_file(path)]
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        boundary_path = self.boundary_dir / f"{image_path.stem}.png"
        if not boundary_path.exists():
            boundary_path = self.boundary_dir / image_path.name

        image = _read_rgb(image_path)
        boundary = _read_grayscale(boundary_path)

        image = _resize(image, self.image_size)
        boundary = _resize(boundary, self.image_size)

        image = image.astype(np.float32) / 255.0
        boundary = (boundary > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        boundary_tensor = torch.from_numpy(boundary).unsqueeze(0).contiguous()
        return image_tensor, boundary_tensor


def build_dataloader(
    image_dir: str,
    boundary_dir: str,
    batch_size: int = 4,
    image_size: Optional[tuple[int, int]] = (256, 256),
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    dataset = SilkwormBoundaryDataset(image_dir=image_dir, boundary_dir=boundary_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _boundary_outputs_to_probs(outputs: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to probabilities in a shape-safe way."""
    if outputs.min().item() < 0.0 or outputs.max().item() > 1.0:
        return torch.sigmoid(outputs)
    return outputs


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def train_boundary_model(
    image_dir: str,
    boundary_dir: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    image_size: tuple[int, int] = (256, 256),
    save_path: str = "boundary_model.pt",
    log_every_batches: int = 0,
) -> AttentionUNet:
    """Train AttentionUNet to predict boundary masks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(image_dir, boundary_dir, batch_size=batch_size, image_size=image_size, shuffle=True)

    model = AttentionUNet(apply_sigmoid=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    train_start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.perf_counter()
        total_batches = max(len(loader), 1)

        for batch_index, (images, targets) in enumerate(loader, start=1):
            batch_start_time = time.perf_counter()
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            loss = 0.5 * bce_loss(logits, targets) + 0.5 * dice_loss(probs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_duration = time.perf_counter() - batch_start_time

            if log_every_batches > 0 and (batch_index % log_every_batches == 0 or batch_index == total_batches):
                avg_batch_time = (time.perf_counter() - epoch_start_time) / batch_index
                remaining_batches = total_batches - batch_index
                epoch_eta_seconds = avg_batch_time * remaining_batches
                print(
                    f"Epoch {epoch + 1:03d}/{epochs:03d} "
                    f"Batch {batch_index:04d}/{total_batches:04d} "
                    f"- loss: {loss.item():.4f} "
                    f"- batch_time: {_format_duration(batch_duration)} "
                    f"- epoch_eta: {_format_duration(epoch_eta_seconds)}"
                )

        mean_loss = running_loss / max(len(loader), 1)
        epoch_duration = time.perf_counter() - epoch_start_time
        elapsed_time = time.perf_counter() - train_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs

        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} "
            f"- loss: {mean_loss:.4f} "
            f"- epoch_time: {_format_duration(epoch_duration)} "
            f"- elapsed: {_format_duration(elapsed_time)} "
            f"- ETA: {_format_duration(eta_seconds)}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": image_size,
            "class_name": model.__class__.__name__,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")
    return model


def check_lesion_on_individual(img: np.ndarray, mask: np.ndarray) -> bool:
    """Lightweight disease-spot heuristic for a single watershed instance."""
    if img is None or mask is None:
        return False

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D single-channel array")

    if cv2.countNonZero(mask) == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    dark_spots = cv2.inRange(hsv, (0, 20, 0), (180, 255, 90))
    lesion_pixels = cv2.bitwise_and(dark_spots, dark_spots, mask=mask)

    spot_count = int(cv2.countNonZero(lesion_pixels))
    object_area = int(cv2.countNonZero(mask))
    threshold = max(25, int(object_area * 0.01))
    return spot_count >= threshold


def process_silkworm_pipeline(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device | str,
    show: bool = False,
) -> tuple[np.ndarray, int]:
    """Run boundary prediction, watershed separation, and lesion inspection."""
    if isinstance(device, str):
        device = torch.device(device)

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        outputs = model(input_tensor)

    boundary_probs = _boundary_outputs_to_probs(outputs).squeeze().detach().cpu().numpy()
    boundary_mask = (boundary_probs > 0.5).astype(np.uint8) * 255

    # Watershed expects object-like regions in the inverted boundary map.
    inverted_boundary = cv2.bitwise_not(boundary_mask)
    distance_map = cv2.distanceTransform(inverted_boundary, cv2.DIST_L2, 5)

    max_distance = float(distance_map.max())
    if max_distance > 0.0:
        _, sure_fg = cv2.threshold(distance_map, 0.4 * max_distance, 255, cv2.THRESH_BINARY)
    else:
        sure_fg = np.zeros_like(distance_map, dtype=np.float32)

    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(inverted_boundary, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = markers.astype(np.int32)

    markers = cv2.watershed(image_bgr.copy(), markers)

    annotated = image_rgb.copy()
    diseased_instances = 0

    for instance_id in np.unique(markers):
        if instance_id in (-1, 1):
            continue

        instance_mask = (markers == instance_id).astype(np.uint8) * 255
        if cv2.countNonZero(instance_mask) == 0:
            continue

        if check_lesion_on_individual(image_rgb, instance_mask):
            diseased_instances += 1
            overlay = np.zeros_like(annotated, dtype=np.uint8)
            overlay[:] = (255, 0, 0)
            instance_region = instance_mask.astype(bool)
            annotated[instance_region] = cv2.addWeighted(
                annotated[instance_region], 0.5,
                overlay[instance_region], 0.5,
                0.0,
            )

    result_path = Path("data/outputs/result_output.png").resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(result_path), result_bgr)
    print(f"Total diseased instances found: {diseased_instances}")
    print(f"Result image saved to: file://{result_path}")

    if show:
        window_name = "Silkworm Segmentation Result"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.imshow(window_name, result_bgr)
        print("Press any key on the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return annotated, diseased_instances


def load_model_checkpoint(checkpoint_path: str, device: torch.device | str) -> AttentionUNet:
    """Load a trained AttentionUNet checkpoint produced by this script."""
    if isinstance(device, str):
        device = torch.device(device)

    model = AttentionUNet(apply_sigmoid=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Silkworm boundary labeling, training, and watershed inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-labels", help="Generate boundary ground truth from binary masks")
    prepare_parser.add_argument("--mask-dir", required=True)
    prepare_parser.add_argument("--output-dir", required=True)

    train_parser = subparsers.add_parser("train", help="Train AttentionUNet on boundary labels")
    train_parser.add_argument("--image-dir", required=True)
    train_parser.add_argument("--boundary-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--height", type=int, default=256)
    train_parser.add_argument("--width", type=int, default=256)
    train_parser.add_argument("--save-path", default="data/checkpoints/boundary_model.pt")
    train_parser.add_argument(
        "--log-every-batches",
        type=int,
        default=0,
        help="Log batch-level timing/loss every N batches (0 disables batch logs)",
    )

    infer_parser = subparsers.add_parser("infer", help="Run watershed inference on a single image")
    infer_parser.add_argument("--image-path", required=True)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--show", action="store_true", help="Show the result in a popup window")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "prepare-labels":
        generate_boundary_labels(args.mask_dir, args.output_dir)
        return

    if args.command == "train":
        train_boundary_model(
            image_dir=args.image_dir,
            boundary_dir=args.boundary_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=(args.height, args.width),
            save_path=args.save_path,
            log_every_batches=args.log_every_batches,
        )
        return

    if args.command == "infer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_checkpoint(args.checkpoint, device)
        process_silkworm_pipeline(args.image_path, model, device, show=args.show)
        return


if __name__ == "__main__":
    main()
