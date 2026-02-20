"""Full pipeline: preprocess → detect → classify → OCR-validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2

from classify import classify_pins_batch, load_classifier
from detect import YOLOModel, crop_detections, detect_pin_diagrams, load_model, sort_detections
from model import PinClassifier
from ocr import cross_validate
from preprocess import rectify_sheet


# ── Model caches (keyed by resolved path string) ──────────────────────────
# Loading YOLO or the CNN from disk takes ~100 ms per call.  When scanning
# multiple sheets in sequence the same weights file is used every time —
# cache the loaded objects so only the first call pays the I/O cost.

@lru_cache(maxsize=4)
def _cached_detector(path_str: str) -> YOLOModel:
    return load_model(Path(path_str))


@lru_cache(maxsize=4)
def _cached_classifier(path_str: str) -> tuple[PinClassifier, "torch.device"]:  # noqa: F821
    return load_classifier(Path(path_str))


@dataclass
class ThrowResult:
    """Result for a single throw (one pin diagram)."""

    column: int
    row: int
    score: int
    pins_down: list[int] = field(default_factory=list)
    confidence: float = 0.0
    classification_confidence: float = 0.0
    ocr_mismatch: bool = False


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
    use_ocr: bool = False,
) -> SheetResult:
    """Full pipeline: load → preprocess → detect → classify → OCR-validate.

    ``use_ocr=False`` (default) skips Tesseract entirely (~12× faster).
    Pass ``use_ocr=True`` to cross-validate CNN scores against printed digits.
    """
    classifier_path = classifier_path or DEFAULT_CLASSIFIER_PATH

    raw = cv2.imread(str(image_path))
    if raw is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    rectified = rectify_sheet(raw)

    yolo: YOLOModel | None = None
    detector_path = model_path or DEFAULT_DETECTOR_PATH
    if detector_path.exists():
        yolo = _cached_detector(str(detector_path.resolve()))

    sorted_dets = sort_detections(detect_pin_diagrams(yolo, rectified, confidence))
    crops = crop_detections(rectified, sorted_dets)

    if not crops:
        return SheetResult()

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found at {classifier_path}. "
            "Train a model first (see `pinsheet-scanner train`)."
        )
    cnn, device = _cached_classifier(str(classifier_path.resolve()))
    classifications = classify_pins_batch(cnn, crops, device=device)

    flagged = set(cross_validate(rectified, sorted_dets, classifications) if use_ocr else [])

    col_indices = {d.column for d in sorted_dets}
    result = SheetResult(
        columns=len(col_indices),
        rows_per_column=max(
            (sum(1 for d in sorted_dets if d.column == c) for c in col_indices),
            default=0,
        ),
    )

    for i, (det, (pins, cls_conf)) in enumerate(zip(sorted_dets, classifications)):
        result.throws.append(ThrowResult(
            column=det.column, row=det.row, score=sum(pins),
            pins_down=pins, confidence=det.confidence,
            classification_confidence=cls_conf, ocr_mismatch=i in flagged,
        ))
    return result
