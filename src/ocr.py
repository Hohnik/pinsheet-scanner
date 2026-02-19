"""OCR cross-validation of CNN pin-count predictions.

Reads the printed score digit to the left of each pin diagram via
Tesseract.  Optional — returns empty results when pytesseract is missing.
"""

from __future__ import annotations

import cv2
import numpy as np

from detect import Detection

_TSR_CFG = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def read_score_adjacent(gray: np.ndarray, detection: Detection) -> int | None:
    """OCR the printed score number to the left of a detected diagram.

    Returns parsed integer (0–9), or ``None`` if unavailable.
    """
    if not _has_tesseract():
        return None
    import pytesseract

    x0 = max(0, int(detection.x_min - detection.width * 1.5))
    x1 = max(0, int(detection.x_min - detection.width * 0.1))
    y0, y1 = max(0, detection.y_min), min(gray.shape[0], detection.y_max)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    scale = max(1, 48 // roi.shape[0])
    up = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(binary, config=_TSR_CFG).strip()
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
    """Return indices where ``sum(cnn_pins) != ocr_score``."""
    return [
        idx for idx, (det, (pins, _)) in enumerate(zip(detections, predictions))
        if (ocr := read_score_adjacent(gray, det)) is not None and sum(pins) != ocr
    ]
