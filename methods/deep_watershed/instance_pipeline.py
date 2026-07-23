"""Instance segmentation pipeline for overlapping silkworms.

This module implements a clean 4-step pipeline:
1) Boundary prediction with Attention U-Net (PyTorch)
2) Distance transform + seed marker generation (OpenCV)
3) Watershed instance separation (OpenCV)
4) Independent lesion inspection per instance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import AttentionUNet, DiceLoss


# -----------------------------------------------------------------------------
# Step 2: Distance Transform & Marker Generation
# -----------------------------------------------------------------------------


@dataclass
class MarkerGenerationResult:
    """Container for intermediate maps used by watershed."""

    boundary_binary: np.ndarray
    silkworm_foreground: np.ndarray
    distance_map: np.ndarray
    seed_markers: np.ndarray


def prepare_watershed_markers(
    boundary_prob: np.ndarray,
    boundary_threshold: float = 0.5,
    peak_threshold_ratio: float = 0.45,
    min_seed_area: int = 20,
) -> MarkerGenerationResult:
    """Generate watershed seeds from a boundary probability map.

    Args:
        boundary_prob: float map in [0, 1], high at contact/boundary pixels.
        boundary_threshold: threshold to create binary boundary.
        peak_threshold_ratio: ratio of max distance used to keep local peaks.
        min_seed_area: removes tiny noisy seed components.

    Returns:
        MarkerGenerationResult containing binary boundary, inverted foreground,
        distance map, and connected-component seed IDs.
    """

    boundary_binary = (boundary_prob >= boundary_threshold).astype(np.uint8) * 255

    # Invert boundary map: body interiors become high-confidence foreground.
    silkworm_foreground = cv2.bitwise_not(boundary_binary)

    # Clean small artifacts to stabilize distance peaks.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    silkworm_foreground = cv2.morphologyEx(silkworm_foreground, cv2.MORPH_OPEN, kernel, iterations=1)

    distance_map = cv2.distanceTransform(silkworm_foreground, cv2.DIST_L2, 5)
    max_distance = float(distance_map.max())

    if max_distance > 0.0:
        _, local_maxima = cv2.threshold(
            distance_map,
            peak_threshold_ratio * max_distance,
            255,
            cv2.THRESH_BINARY,
        )
    else:
        local_maxima = np.zeros_like(distance_map, dtype=np.uint8)

    local_maxima = local_maxima.astype(np.uint8)

    # Remove tiny maxima components before assigning IDs.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local_maxima, connectivity=8)
    filtered_maxima = np.zeros_like(local_maxima)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_seed_area:
            filtered_maxima[labels == label_id] = 255

    _, seed_markers = cv2.connectedComponents(filtered_maxima, connectivity=8)
    seed_markers = seed_markers.astype(np.int32)

    return MarkerGenerationResult(
        boundary_binary=boundary_binary,
        silkworm_foreground=silkworm_foreground,
        distance_map=distance_map,
        seed_markers=seed_markers,
    )


# -----------------------------------------------------------------------------
# Step 3: Watershed Separation
# -----------------------------------------------------------------------------


def run_watershed_separation(
    rgb_image: np.ndarray,
    boundary_binary: np.ndarray,
    seed_markers: np.ndarray,
) -> np.ndarray:
    """Run watershed using prepared markers.

    Marker policy (required):
    - boundary pixels -> 0 (barriers)
    - background -> 1
    - each local-maxima seed -> unique ID >= 2
    """

    if rgb_image.dtype != np.uint8:
        raise ValueError("rgb_image must be uint8")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape [H, W, 3]")

    # Base map: everything starts as background = 1.
    watershed_markers = np.ones_like(seed_markers, dtype=np.int32)

    # Keep unique instance seed IDs by offsetting from 2 upward.
    seed_mask = seed_markers > 0
    watershed_markers[seed_mask] = seed_markers[seed_mask] + 1

    # Force AI boundary predictions as unknown barriers.
    watershed_markers[boundary_binary > 0] = 0

    # OpenCV watershed writes labels in-place and uses -1 for final borders.
    watershed_output = cv2.watershed(rgb_image.copy(), watershed_markers)
    return watershed_output


# -----------------------------------------------------------------------------
# Step 4: Independent Lesion Inspection
# -----------------------------------------------------------------------------


def inspect_lesion(instance_mask: np.ndarray, rgb_image: np.ndarray) -> Dict[str, float | bool]:
    """Placeholder lesion inspection isolated to one silkworm instance.

    The check is intentionally simple:
    - Convert to HSV
    - Detect very dark pixels as candidate disease spots
    - Report ratio of dark pixels inside this instance only
    """

    if instance_mask.dtype != np.uint8:
        instance_mask = instance_mask.astype(np.uint8)

    if cv2.countNonZero(instance_mask) == 0:
        return {"is_diseased": False, "dark_ratio": 0.0, "dark_pixels": 0.0}

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Simple dark-spot detector (tunable placeholder).
    dark_pixels = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    dark_pixels = cv2.bitwise_and(dark_pixels, dark_pixels, mask=instance_mask)

    dark_count = float(cv2.countNonZero(dark_pixels))
    area = float(cv2.countNonZero(instance_mask))
    dark_ratio = dark_count / max(area, 1.0)

    # Conservative placeholder decision threshold.
    is_diseased = dark_ratio > 0.015

    return {
        "is_diseased": bool(is_diseased),
        "dark_ratio": float(dark_ratio),
        "dark_pixels": dark_count,
    }


def inspect_all_instances(instance_map: np.ndarray, rgb_image: np.ndarray) -> List[Dict[str, float | int | bool]]:
    """Loop over all watershed instance IDs and inspect each independently."""

    results: List[Dict[str, float | int | bool]] = []

    # In watershed output: -1 is border; 1 is background in this setup.
    candidate_ids = [iid for iid in np.unique(instance_map) if iid > 1]

    for instance_id in candidate_ids:
        instance_mask = (instance_map == instance_id).astype(np.uint8) * 255
        lesion_report = inspect_lesion(instance_mask, rgb_image)

        results.append(
            {
                "instance_id": int(instance_id),
                "pixel_area": int(cv2.countNonZero(instance_mask)),
                "is_diseased": bool(lesion_report["is_diseased"]),
                "dark_ratio": float(lesion_report["dark_ratio"]),
            }
        )

    return results


# -----------------------------------------------------------------------------
# End-to-end pipeline utility
# -----------------------------------------------------------------------------


def predict_boundary_map(
    model: nn.Module,
    rgb_image: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    """Run Step 1 model inference and return HxW boundary probability map."""

    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    image_tensor = torch.from_numpy(rgb_image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        boundary_prob = model(image_tensor)

    boundary_prob = boundary_prob.squeeze().detach().cpu().numpy()
    return boundary_prob.astype(np.float32)


def run_full_pipeline(
    rgb_image: np.ndarray,
    model: nn.Module,
    device: torch.device | str = "cpu",
    boundary_threshold: float = 0.5,
) -> Dict[str, np.ndarray | List[Dict[str, float | int | bool]]]:
    """Execute all 4 steps and return intermediate + final results."""

    boundary_prob = predict_boundary_map(model=model, rgb_image=rgb_image, device=device)

    marker_data = prepare_watershed_markers(
        boundary_prob=boundary_prob,
        boundary_threshold=boundary_threshold,
    )

    instance_map = run_watershed_separation(
        rgb_image=rgb_image,
        boundary_binary=marker_data.boundary_binary,
        seed_markers=marker_data.seed_markers,
    )

    lesion_reports = inspect_all_instances(instance_map=instance_map, rgb_image=rgb_image)

    return {
        "boundary_prob": boundary_prob,
        "boundary_binary": marker_data.boundary_binary,
        "distance_map": marker_data.distance_map,
        "seed_markers": marker_data.seed_markers,
        "instance_map": instance_map,
        "lesion_reports": lesion_reports,
    }


# -----------------------------------------------------------------------------
# Mock pipeline execution (dummy arrays) for data-flow demonstration
# -----------------------------------------------------------------------------


def _create_dummy_silkworm_like_image(height: int = 256, width: int = 384) -> np.ndarray:
    """Create a synthetic RGB frame with overlapping worm-like blobs + dark spots."""

    image = np.full((height, width, 3), 225, dtype=np.uint8)

    # Draw elongated overlapping bodies.
    cv2.ellipse(image, (120, 130), (70, 28), 12, 0, 360, (198, 204, 170), -1)
    cv2.ellipse(image, (185, 145), (75, 30), -8, 0, 360, (195, 200, 165), -1)
    cv2.ellipse(image, (255, 115), (65, 26), 20, 0, 360, (202, 210, 175), -1)

    # Add a few dark candidate lesions.
    cv2.circle(image, (178, 140), 5, (35, 35, 35), -1)
    cv2.circle(image, (260, 110), 4, (45, 45, 45), -1)

    return image


def mock_pipeline_execution() -> None:
    """Demonstrate full data flow from Step 1 to Step 4 using dummy inputs."""

    device = torch.device("cpu")

    # Randomly initialized model is enough for shape/data-flow demonstration.
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=32)

    dummy_rgb = _create_dummy_silkworm_like_image()

    outputs = run_full_pipeline(
        rgb_image=dummy_rgb,
        model=model,
        device=device,
        boundary_threshold=0.5,
    )

    print("==== Mock Pipeline Execution ====")
    print(f"Input RGB shape: {dummy_rgb.shape}")
    print(f"Boundary prob shape: {outputs['boundary_prob'].shape}")
    print(f"Boundary binary shape: {outputs['boundary_binary'].shape}")
    print(f"Distance map max: {float(np.max(outputs['distance_map'])):.4f}")
    print(f"Seed marker IDs: {np.unique(outputs['seed_markers'])[:10]}")
    print(f"Instance IDs (watershed): {np.unique(outputs['instance_map'])[:12]}")

    lesion_reports = outputs["lesion_reports"]
    print(f"Instances inspected: {len(lesion_reports)}")
    for report in lesion_reports[:5]:
        print(report)


if __name__ == "__main__":
    mock_pipeline_execution()
