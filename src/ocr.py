"""OCR cross-validation of CNN pin-count predictions.

Reads the printed score digit adjacent to each pin diagram via Tesseract.
Disabled by default in the scan pipeline (pass ``use_ocr=True`` to enable).

Performance note: the naive approach calls tesseract once per diagram
(90 subprocess spawns ~= 5 s/sheet).  ``cross_validate`` instead stitches
all score regions into a single image and calls tesseract once, reducing
OCR time to ~100 ms/sheet.
"""

from __future__ import annotations

import cv2
import numpy as np

from detect import Detection

# Single-digit whitelist; PSM 7 = single text line per call (legacy),
# PSM 6 = assume uniform block of text (batch call).
_TSR_CFG_BATCH = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _score_roi(gray: np.ndarray, det: Detection) -> np.ndarray | None:
    """Extract and upscale the score region to the left of a detection."""
    x0 = max(0, int(det.x_min - det.width * 1.5))
    x1 = max(0, int(det.x_min - det.width * 0.1))
    y0 = max(0, det.y_min)
    y1 = min(gray.shape[0], det.y_max)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    scale = max(1, 48 // roi.shape[0])
    up = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale),
                    interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def cross_validate(
    gray: np.ndarray,
    detections: list[Detection],
    predictions: list[tuple[list[int], float]],
) -> list[int]:
    """Return indices where ``sum(cnn_pins) != ocr_score``.

    All score regions are stitched into a single image and Tesseract is called
    **once** per sheet instead of once per diagram.  Falls back to an empty
    list when pytesseract / tesseract is unavailable.
    """
    if not _has_tesseract():
        return []
    import pytesseract

    # --- Extract ROIs; track which detection index each belongs to ----------
    rois: list[np.ndarray] = []
    det_indices: list[int] = []
    for idx, det in enumerate(detections):
        roi = _score_roi(gray, det)
        if roi is not None:
            rois.append(roi)
            det_indices.append(idx)

    if not rois:
        return []

    # --- Normalise heights and stack vertically with separator rows ----------
    target_h = max(r.shape[0] for r in rois)
    max_w    = max(r.shape[1] for r in rois)
    sep_h    = max(4, target_h // 6)   # white gap between rows

    strips: list[np.ndarray] = []
    for roi in rois:
        pad = target_h - roi.shape[0]
        top, bot = pad // 2, pad - pad // 2
        padded = cv2.copyMakeBorder(roi, top, bot, 0, max_w - roi.shape[1],
                                    cv2.BORDER_CONSTANT, value=255)
        strips.append(padded)
        strips.append(np.full((sep_h, max_w), 255, dtype=np.uint8))

    stacked = np.vstack(strips)

    # --- One tesseract call for the whole sheet ------------------------------
    try:
        raw = pytesseract.image_to_string(stacked, config=_TSR_CFG_BATCH)
    except Exception:
        return []

    # Parse non-empty lines; tesseract may produce fewer lines than rois if
    # it merges or skips regions, so zip safely.
    ocr_digits: list[int | None] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digits = "".join(c for c in stripped if c.isdigit())
        if digits:
            val = int(digits[-1])   # take last digit (handles e.g. "9\n9")
            ocr_digits.append(val if 0 <= val <= 9 else None)
        else:
            ocr_digits.append(None)

    # --- Cross-reference with CNN predictions --------------------------------
    flagged: list[int] = []
    for (det_idx, ocr_val) in zip(det_indices, ocr_digits):
        if ocr_val is None:
            continue
        pins, _ = predictions[det_idx]
        if sum(pins) != ocr_val:
            flagged.append(det_idx)

    return flagged
