"""Debug script to save cropped pin diagrams and run CNN classification on each.

Usage:
    uv run python -m scripts.debug_crops pinsheet_example.jpeg
    uv run python -m scripts.debug_crops pinsheet_example.jpeg --output debug_crops
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from pinsheet_scanner.classify import (
    classify_pins_batch_with_confidence,
    load_classifier,
)
from pinsheet_scanner.detect import (
    crop_detections,
    detect_pin_diagrams,
    draw_detections,
    load_model,
    sort_detections,
)
from pinsheet_scanner.pipeline import DEFAULT_CLASSIFIER_PATH, DEFAULT_DETECTOR_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save cropped pin diagrams and run CNN classification for debugging.",
    )
    parser.add_argument(
        "image", type=Path, help="Path to the scanned score sheet image."
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="YOLO detector weights (.pt)."
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=None,
        help="CNN classifier weights (.pt).",
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
        help="Output directory (default: debug_crops/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model or DEFAULT_DETECTOR_PATH
    classifier_path = args.classifier_model or DEFAULT_CLASSIFIER_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Detector weights not found at {model_path}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found at {args.image}")

    raw_dir = args.output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect, sort, crop
    sorted_dets = sort_detections(
        detect_pin_diagrams(
            load_model(model_path), image, confidence_threshold=args.confidence
        )
    )
    print(f"Detected {len(sorted_dets)} pin diagrams")

    cv2.imwrite(
        str(args.output / "annotated_full.jpg"), draw_detections(image, sorted_dets)
    )
    print(f"Saved annotated image to {args.output / 'annotated_full.jpg'}")

    crops = crop_detections(gray, sorted_dets)
    names = [f"c{d.column:02d}_r{d.row:02d}" for d in sorted_dets]
    for name, crop in zip(names, crops):
        cv2.imwrite(str(raw_dir / f"{name}.png"), crop)

    # Classify (if weights exist)
    has_classifier = classifier_path.exists()
    classifications: list[tuple[list[int], float]] = []
    if has_classifier:
        cnn, device = load_classifier(classifier_path)
        classifications = classify_pins_batch_with_confidence(cnn, crops, device=device)
    else:
        print(
            f"\nClassifier weights not found at {classifier_path} — skipping classification."
        )

    # Print results
    header = f"{'Name':<20} {'Size':<12}"
    if has_classifier:
        header += f" {'Pins':>10} {'Score':>6} {'Conf':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    csv_rows: list[dict[str, str | int | float]] = []
    for i, (det, crop) in enumerate(zip(sorted_dets, crops)):
        row_data: dict[str, str | int | float] = {
            "name": names[i],
            "width": crop.shape[1],
            "height": crop.shape[0],
            "column": det.column,
            "row": det.row,
            "det_confidence": round(det.confidence, 4),
        }
        line = f"{names[i]:<20} {crop.shape[1]}x{crop.shape[0]:<12}"

        if has_classifier and i < len(classifications):
            pins, cls_conf = classifications[i]
            pins_str = "".join(str(p) for p in pins)
            line += f" {pins_str:>10} {sum(pins):>6} {cls_conf:>6.2f}"
            row_data.update(
                pins=pins_str, score=sum(pins), cls_confidence=round(cls_conf, 4)
            )

        print(line)
        csv_rows.append(row_data)

    # Write CSV
    if csv_rows:
        csv_path = args.output / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nPredictions saved to {csv_path}")

    print(f"\nAll crops saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
