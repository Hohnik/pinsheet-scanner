"""OCR cross-validation of CNN pin-count predictions.

Each detected pin diagram has a printed score number directly to its left
on the score sheet.  Reading that number provides a free checksum:
if ``sum(cnn_pins) != ocr_score`` the prediction is likely wrong.

``pytesseract`` (and the system ``tesseract`` binary) are optional.
All functions return ``None`` or an empty list when unavailable, so the
rest of the pipeline degrades gracefully.

Install the optional dependency with::

    uv sync --extra ocr      # or: pip install pytesseract
    brew install tesseract   # macOS
"""

from __future__ import annotations

import cv2
import numpy as np

from detect import Detection

# Tesseract config: single text line, digits only.
_TSR_CFG = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _upscale_roi(roi: np.ndarray, target_height: int = 48) -> np.ndarray:
    """Upscale and binarise a tiny region for better OCR accuracy."""
    if roi.size == 0:
        return roi
    scale = max(1, target_height // roi.shape[0])
    up = cv2.resize(
        roi,
        (roi.shape[1] * scale, roi.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def read_score_adjacent(
    gray: np.ndarray,
    detection: Detection,
    *,
    left_factor: float = 1.5,
    right_factor: float = 0.1,
) -> int | None:
    """OCR the printed score number to the left of a detected pin diagram.

    Args:
        gray: Full-sheet grayscale image (after rectification).
        detection: The detected pin diagram bounding box.
        left_factor: How many diagram-widths to look left of the diagram.
        right_factor: How many diagram-widths past the left edge to stop.

    Returns:
        Parsed integer score (0–9), or ``None`` if OCR is unavailable or
        the region contains no recognisable digit.
    """
    if not _has_tesseract():
        return None

    import pytesseract

    x0 = max(0, int(detection.x_min - detection.width * left_factor))
    x1 = max(0, int(detection.x_min - detection.width * right_factor))
    y0 = max(0, detection.y_min)
    y1 = min(gray.shape[0], detection.y_max)

    if x1 <= x0 or y1 <= y0:
        return None

    roi = _upscale_roi(gray[y0:y1, x0:x1])
    try:
        text = pytesseract.image_to_string(roi, config=_TSR_CFG).strip()
    except Exception:
        return None

    digits = "".join(c for c in text if c.isdigit())
    if not digits:
        return None
    val = int(digits)
    return val if 0 <= val <= 9 else None


def cross_validate(
    gray: np.ndarray,
    detections: list[Detection],
    predictions: list[tuple[list[int], float]],
) -> list[int]:
    """Return indices of throws where the OCR score disagrees with sum(pins).

    Requires ``pytesseract`` to be installed; returns an empty list otherwise.

    Args:
        gray: Full-sheet grayscale image.
        detections: Detected pin diagram bounding boxes.
        predictions: CNN ``(pins, confidence)`` pairs, one per detection.

    Returns:
        0-based indices where ``sum(cnn_pins) != ocr_score``.
    """
    flagged: list[int] = []
    for idx, (det, (pins, _)) in enumerate(zip(detections, predictions)):
        ocr = read_score_adjacent(gray, det)
        if ocr is not None and sum(pins) != ocr:
            flagged.append(idx)
    return flagged
