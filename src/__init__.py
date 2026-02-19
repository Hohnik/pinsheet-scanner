"""Pinsheet Scanner — extract 9-pin bowling scores from scanned sheets.

Quick start::

    from pinsheet_scanner import process_sheet

    result = process_sheet(Path("sheet.jpeg"))
    for throw in result.throws:
        print(f"Column {throw.column}, Row {throw.row}: {throw.score} pins")

For classification only::

    from pinsheet_scanner import load_classifier, classify_pins_batch_with_confidence

    model, device = load_classifier(Path("models/pin_classifier.pt"))
    results = classify_pins_batch_with_confidence(model, crops, device=device)
"""

from classify import (
    AnyClassifier,
    classify_pins_batch_with_confidence,
    load_classifier,
    preprocess_crop,
)
from constants import CLASSIFIER_INPUT_SIZE, NUM_PINS, PATCH_SIZE, PIN_COORDS_64
from detect import Detection, crop_detections, detect_pin_diagrams, sort_detections
from pipeline import SheetResult, ThrowResult, process_sheet
from preprocess import rectify_sheet

__all__ = [
    # Pipeline
    "process_sheet",
    "SheetResult",
    "ThrowResult",
    # Classification
    "load_classifier",
    "classify_pins_batch_with_confidence",
    "preprocess_crop",
    "AnyClassifier",
    # Detection
    "Detection",
    "detect_pin_diagrams",
    "sort_detections",
    "crop_detections",
    # Preprocessing
    "rectify_sheet",
    # Constants
    "NUM_PINS",
    "CLASSIFIER_INPUT_SIZE",
    "PATCH_SIZE",
    "PIN_COORDS_64",
]
