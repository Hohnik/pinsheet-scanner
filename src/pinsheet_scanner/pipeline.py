"""Full processing pipeline: detect → sort → crop → classify → validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from pinsheet_scanner.classify import pins_from_diagram
from pinsheet_scanner.detect import (
    crop_detections,
    detect_pin_diagrams,
    draw_detections,
    load_model,
    sort_detections,
)


@dataclass
class ThrowResult:
    """Result for a single throw (one pin diagram)."""

    column: int
    row: int
    score: int
    pins_down: list[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SheetResult:
    """Result for an entire score sheet."""

    throws: list[ThrowResult] = field(default_factory=list)
    columns: int = 0
    rows_per_column: int = 0

    @property
    def total_pins(self) -> int:
        return sum(sum(t.pins_down) for t in self.throws)


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "pin_diagram.pt"
)


def process_sheet(
    image_path: Path,
    model_path: Path | None = None,
    confidence: float = 0.25,
    debug: bool = False,
) -> SheetResult:
    """Full pipeline: load image → detect diagrams → sort → classify pins.

    Args:
        image_path: Path to the scanned score sheet image.
        model_path: Path to the trained YOLO model weights (.pt).
            Falls back to ``models/pin_diagram.pt`` relative to the project root.
        confidence: Minimum detection confidence threshold.
        debug: If True, display an annotated image with detections.

    Returns:
        SheetResult containing all classified throws in reading order.
    """
    # Resolve model path
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Train a model first (see scripts/train.py) or pass --model."
        )

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load YOLO model & run detection
    model = load_model(model_path)
    detections = detect_pin_diagrams(model, image, confidence_threshold=confidence)

    # Sort into reading order (left→right columns, top→bottom rows)
    sorted_dets = sort_detections(detections)

    # Debug visualisation
    if debug:
        annotated = draw_detections(image, sorted_dets)
        cv2.imshow("Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Crop each detection from the grayscale image
    crops = crop_detections(gray, sorted_dets)

    # Build the result
    # Determine number of unique columns
    col_indices = {d.column for d in sorted_dets}
    rows_per_col = max(
        (sum(1 for d in sorted_dets if d.column == c) for c in col_indices),
        default=0,
    )

    result = SheetResult(
        columns=len(col_indices),
        rows_per_column=rows_per_col,
    )

    for det, crop in zip(sorted_dets, crops):
        pins = pins_from_diagram(crop)

        throw = ThrowResult(
            column=det.column,
            row=det.row,
            score=sum(pins),
            pins_down=pins,
            confidence=det.confidence,
        )
        result.throws.append(throw)

    return result
