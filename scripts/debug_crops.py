"""Debug script to save cropped pin diagrams so we can inspect what the classifier sees.

Usage:
    uv run python -m scripts.debug_crops pinsheet_example.jpeg

This will create a debug_crops/ directory with:
  - Each cropped diagram as a separate image (named by column and row)
  - An annotated version of the full image with detection boxes drawn
  - A cleaned/resized version of each crop (what the classifier actually processes)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pinsheet_scanner.classify import (
    DEFAULT_MASK_RADIUS,
    DEFAULT_SIZE,
    PIN_POSITIONS,
    build_pin_masks,
    clean_diagram,
    measure_pin_intensity,
    pins_from_diagram,
    resize_diagram,
)
from pinsheet_scanner.detect import (
    crop_detections,
    detect_pin_diagrams,
    draw_detections,
    load_model,
    sort_detections,
)
from pinsheet_scanner.pipeline import DEFAULT_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save cropped pin diagrams for debugging the classifier.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the scanned score sheet image.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a trained YOLO model (.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_crops"),
        help="Output directory for debug images (default: debug_crops/).",
    )
    return parser.parse_args()


def draw_pin_overlay(
    resized: np.ndarray,
    masks: list[np.ndarray],
    intensities: list[float],
    pins: list[int],
    scale: int = 8,
) -> np.ndarray:
    """Draw an enlarged version of the resized diagram with pin positions and
    intensity values overlaid for visual inspection."""
    h, w = resized.shape[:2]
    big = cv2.resize(resized, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    big_bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    for i, (px, py) in enumerate(PIN_POSITIONS):
        cx = int(px * w * scale)
        cy = int(py * h * scale)
        r = DEFAULT_MASK_RADIUS * scale

        # Green = knocked down, Red = standing
        color = (0, 255, 0) if pins[i] == 1 else (0, 0, 255)
        cv2.circle(big_bgr, (cx, cy), r, color, 2)

        label = f"{intensities[i]:.2f}"
        cv2.putText(
            big_bgr,
            label,
            (cx - r, cy - r - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

        # Pin index
        cv2.putText(
            big_bgr,
            str(i),
            (cx - 4, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
        )

    return big_bgr


def main() -> None:
    args = parse_args()

    image_path: Path = args.image
    model_path: Path | None = args.model
    output_dir: Path = args.output

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Create output directories
    raw_dir = output_dir / "raw"
    cleaned_dir = output_dir / "cleaned"
    resized_dir = output_dir / "resized"
    overlay_dir = output_dir / "overlay"
    for d in [raw_dir, cleaned_dir, resized_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect and sort
    model = load_model(model_path)
    detections = detect_pin_diagrams(model, image, confidence_threshold=args.confidence)
    sorted_dets = sort_detections(detections)

    print(f"Detected {len(sorted_dets)} pin diagrams")

    # Save annotated full image
    annotated = draw_detections(image, sorted_dets)
    cv2.imwrite(str(output_dir / "annotated_full.jpg"), annotated)
    print(f"Saved annotated image to {output_dir / 'annotated_full.jpg'}")

    # Crop and save each detection
    crops = crop_detections(gray, sorted_dets)
    masks = build_pin_masks(DEFAULT_SIZE, DEFAULT_MASK_RADIUS)

    print(f"\n{'Name':<20} {'Size':<12} {'Pins':>10} {'Score':>6} {'Intensities'}")
    print("-" * 100)

    for det, crop in zip(sorted_dets, crops):
        name = f"c{det.column:02d}_r{det.row:02d}"

        # Save raw crop
        cv2.imwrite(str(raw_dir / f"{name}.png"), crop)

        # Save cleaned version
        cleaned = clean_diagram(crop)
        cv2.imwrite(str(cleaned_dir / f"{name}.png"), cleaned)

        # Save resized version
        resized = resize_diagram(cleaned, DEFAULT_SIZE)
        cv2.imwrite(str(resized_dir / f"{name}.png"), resized)

        # Compute intensities and classification
        intensities = measure_pin_intensity(resized, masks)
        pins = pins_from_diagram(crop)
        score = sum(pins)

        # Save overlay with pin positions
        overlay = draw_pin_overlay(resized, masks, intensities, pins)
        cv2.imwrite(str(overlay_dir / f"{name}.png"), overlay)

        # Print summary
        pins_str = "".join(str(p) for p in pins)
        int_str = " ".join(f"{v:.2f}" for v in intensities)
        size_str = f"{crop.shape[1]}x{crop.shape[0]}"
        print(f"{name:<20} {size_str:<12} {pins_str:>10} {score:>6} [{int_str}]")

    print(f"\nAll crops saved to {output_dir.resolve()}")
    print("  raw/       - original grayscale crops from the image")
    print("  cleaned/   - after binarization + morphological cleaning")
    print(f"  resized/   - after resizing to {DEFAULT_SIZE}")
    print("  overlay/   - enlarged with pin positions + intensity values")


if __name__ == "__main__":
    main()
