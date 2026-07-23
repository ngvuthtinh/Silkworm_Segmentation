"""Interactive multi-image inference viewer for silkworm boundary pipeline.

Usage example:
python -m methods.deep_watershed.interactive_viewer \
  --image-dir "data/Silkworm Diseases.v1i.yolo26/test/images" \
  --checkpoint data/checkpoints/boundary_model.pt

Controls in viewer window:
- n / d / Right Arrow / Space: next image
- p / a / Left Arrow: previous image
- s: save current overlay image into output directory
- q / ESC: quit
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from .boundary_pipeline import (
    _boundary_outputs_to_probs,
    _is_image_file,
    check_lesion_on_individual,
    load_model_checkpoint,
)


def _collect_images(image_dir: Path, recursive: bool) -> list[Path]:
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    image_paths = [path for path in iterator if path.is_file() and _is_image_file(path)]
    image_paths.sort()
    return image_paths


def _resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    candidate = Path(checkpoint_arg)
    if candidate.exists():
        return candidate

    # Friendly fallback: allow users to pass only the checkpoint filename.
    fallback = Path("data/checkpoints") / candidate.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"checkpoint file not found: {checkpoint_arg}. "
        f"Tried: {candidate} and {fallback}"
    )


def _predict_overlay(image_path: Path, model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, int]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
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
                annotated[instance_region],
                0.5,
                overlay[instance_region],
                0.5,
                0.0,
            )

    return annotated, diseased_instances


def _draw_header(image_rgb: np.ndarray, title: str, disease_count: int) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    header_h = 70
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_h), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"diseased_instances: {disease_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _save_visual(output_dir: Path, image_path: Path, annotated_bgr: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_annotated.png"
    cv2.imwrite(str(out_path), annotated_bgr)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive inference viewer for multiple images")
    parser.add_argument("--image-dir", required=True, help="Directory containing test images")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", default="data/outputs/interactive", help="Directory for saved overlays")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images under image-dir")
    parser.add_argument("--start-index", type=int, default=0, help="Index of first image to show")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use terminal-only navigation mode (no GUI window)",
    )
    parser.add_argument(
        "--random-one",
        action="store_true",
        help="Pick one random image, run inference once, save output, then exit",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    image_dir = Path(args.image_dir)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"image-dir is not a valid directory: {image_dir}")

    image_paths = _collect_images(image_dir, recursive=args.recursive)
    if not image_paths:
        raise ValueError(f"No image files found in {image_dir}")

    total = len(image_paths)
    index = max(0, min(args.start_index, total - 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(str(checkpoint_path), device)

    if args.random_one:
        image_path = random.choice(image_paths)
        annotated_rgb, diseased_count = _predict_overlay(image_path, model, device)
        frame = _draw_header(annotated_rgb, f"[random] {image_path.name}", diseased_count)
        out_path = _save_visual(output_dir, image_path, frame)
        print(f"Random image: {image_path}")
        print(f"diseased_instances={diseased_count}")
        print(f"Saved: {out_path}")
        return

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless_mode = args.headless or not has_display

    if headless_mode:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Running in headless mode (no display detected).")
        print("Controls in terminal: n=next, p=prev, s=save current, q=quit")

        while True:
            image_path = image_paths[index]
            annotated_rgb, diseased_count = _predict_overlay(image_path, model, device)
            frame = _draw_header(annotated_rgb, f"[{index + 1}/{total}] {image_path.name}", diseased_count)

            preview_path = output_dir / "_current_preview.png"
            cv2.imwrite(str(preview_path), frame)
            print(
                f"Viewing {index + 1}/{total}: {image_path} | "
                f"diseased_instances={diseased_count} | preview={preview_path}"
            )

            cmd = input("[n]ext [p]rev [s]ave [q]uit > ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd in ("n", "", "next"):
                index = (index + 1) % total
                continue
            if cmd in ("p", "prev", "previous"):
                index = (index - 1) % total
                continue
            if cmd in ("s", "save"):
                out_path = _save_visual(output_dir, image_path, frame)
                print(f"Saved: {out_path}")
                continue

            print("Unknown command. Use n, p, s, or q.")

        return

    window_name = "Silkworm Interactive Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Controls: n/d/Right/Space=next, p/a/Left=prev, s=save, q/ESC=quit")

    while True:
        image_path = image_paths[index]
        annotated_rgb, diseased_count = _predict_overlay(image_path, model, device)
        title = f"[{index + 1}/{total}] {image_path.name}"
        frame = _draw_header(annotated_rgb, title, diseased_count)

        cv2.imshow(window_name, frame)
        print(f"Viewing {index + 1}/{total}: {image_path} | diseased_instances={diseased_count}")

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord("d"), ord(" "), 83, 2555904):
            index = (index + 1) % total
            continue
        if key in (ord("p"), ord("a"), 81, 2424832):
            index = (index - 1) % total
            continue
        if key == ord("s"):
            out_path = _save_visual(output_dir, image_path, frame)
            print(f"Saved: {out_path}")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
