"""Full processing pipeline: detect → sort → crop → classify.

Uses YOLO for pin diagram detection and a tiny CNN for pin state
classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class ThrowResult:
    """Result for a single throw (one pin diagram)."""

    column: int
    row: int
    score: int
    pins_down: list[int] = field(default_factory=list)
    confidence: float = 0.0
    classification_confidence: float = 0.0


@dataclass
class SheetResult:
    """Result for an entire score sheet."""

    throws: list[ThrowResult] = field(default_factory=list)
    columns: int = 0
    rows_per_column: int = 0

    @property
    def total_pins(self) -> int:
        return sum(sum(t.pins_down) for t in self.throws)


DEFAULT_DETECTOR_PATH = Path("models/pin_diagram.pt")
DEFAULT_CLASSIFIER_PATH = Path("models/pin_classifier.pt")


def process_sheet(
    image_path: Path,
    model_path: Path | None = None,
    classifier_path: Path | None = None,
    confidence: float = 0.25,
    debug: bool = False,
) -> SheetResult:
    """Full pipeline: load image → detect diagrams → sort → classify pins.

    Args:
        image_path: Path to the scanned score sheet image.
        model_path: Path to the trained YOLO detector weights (.pt).
            Falls back to ``models/pin_diagram.pt``.
        classifier_path: Path to the trained CNN classifier weights (.pt).
            Falls back to ``models/pin_classifier.pt``.
        confidence: Minimum detection confidence threshold.
        debug: If True, display an annotated image with detections.

    Returns:
        SheetResult containing all classified throws in reading order.
    """
    model_path = model_path or DEFAULT_DETECTOR_PATH
    classifier_path = classifier_path or DEFAULT_CLASSIFIER_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Train a model first (see `pinsheet-scanner train-detector`) or pass --model."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect → sort → crop
    detector = load_model(model_path)
    sorted_dets = sort_detections(
        detect_pin_diagrams(detector, image, confidence_threshold=confidence)
    )

    if debug:
        cv2.imshow("Detections", draw_detections(image, sorted_dets))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    crops = crop_detections(gray, sorted_dets)

    # Classify
    cnn, device = load_classifier(classifier_path)
    classifications = classify_pins_batch_with_confidence(cnn, crops, device=device)

    # Assemble results
    col_indices = {d.column for d in sorted_dets}
    result = SheetResult(
        columns=len(col_indices),
        rows_per_column=max(
            (sum(1 for d in sorted_dets if d.column == c) for c in col_indices),
            default=0,
        ),
    )

    for det, (pins, cls_conf) in zip(sorted_dets, classifications):
        result.throws.append(
            ThrowResult(
                column=det.column,
                row=det.row,
                score=sum(pins),
                pins_down=pins,
                confidence=det.confidence,
                classification_confidence=cls_conf,
            )
        )

    return result
