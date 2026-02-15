"""Full processing pipeline: detect → sort → crop → classify → validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from pinsheet_scanner.classify import pins_and_intensities_from_diagram
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
    classification_confidence: float = 0.0
    expected_score: int | None = None


@dataclass
class SheetResult:
    """Result for an entire score sheet."""

    throws: list[ThrowResult] = field(default_factory=list)
    columns: int = 0
    rows_per_column: int = 0

    @property
    def total_pins(self) -> int:
        return sum(sum(t.pins_down) for t in self.throws)

    @property
    def mismatches(self) -> list[ThrowResult]:
        """Return throws where classified score differs from OCR expected score."""
        return [
            t
            for t in self.throws
            if t.expected_score is not None and t.score != t.expected_score
        ]


DEFAULT_MODEL_PATH = Path("models/pin_diagram.pt")


def calculate_classification_confidence(
    intensities: list[float], pins: list[int]
) -> float:
    """Calculate confidence as gap between highest standing and lowest knocked down intensity.

    A larger gap means higher confidence in the classification.
    If all pins are in the same state, confidence is 1.0 (maximum).
    """
    if not intensities or not pins or len(intensities) != len(pins):
        return 0.0

    knocked_down = [intensities[i] for i in range(len(pins)) if pins[i] == 1]
    standing = [intensities[i] for i in range(len(pins)) if pins[i] == 0]

    if not knocked_down or not standing:
        return 1.0

    min_down = min(knocked_down)
    max_standing = max(standing)

    gap = min_down - max_standing
    return max(0.0, min(1.0, gap))


def process_sheet(
    image_path: Path,
    model_path: Path | None = None,
    confidence: float = 0.25,
    debug: bool = False,
    ocr: bool = False,
) -> SheetResult:
    """Full pipeline: load image → detect diagrams → sort → classify pins.

    Args:
        image_path: Path to the scanned score sheet image.
        model_path: Path to the trained YOLO model weights (.pt).
            Falls back to ``models/pin_diagram.pt`` relative to the project root.
        confidence: Minimum detection confidence threshold.
        debug: If True, display an annotated image with detections.
        ocr: If True, extract printed score digits via OCR for validation.

    Returns:
        SheetResult containing all classified throws in reading order.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Train a model first (see scripts/train.py) or pass --model."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    model = load_model(model_path)
    detections = detect_pin_diagrams(model, image, confidence_threshold=confidence)

    sorted_dets = sort_detections(detections)

    if debug:
        annotated = draw_detections(image, sorted_dets)
        cv2.imshow("Detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    crops = crop_detections(gray, sorted_dets)

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
        pins, intensities = pins_and_intensities_from_diagram(crop)
        class_conf = calculate_classification_confidence(intensities, pins)

        throw = ThrowResult(
            column=det.column,
            row=det.row,
            score=sum(pins),
            pins_down=pins,
            confidence=det.confidence,
            classification_confidence=class_conf,
        )
        result.throws.append(throw)

    if ocr and sorted_dets:
        from pinsheet_scanner.ocr import extract_row_scores

        ocr_scores = extract_row_scores(gray, sorted_dets)
        for throw, expected in zip(result.throws, ocr_scores):
            throw.expected_score = expected

    return result
