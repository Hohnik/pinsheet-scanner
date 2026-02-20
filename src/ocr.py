"""OCR cross-validation of CNN pin-count predictions.

Reads the printed score digit adjacent to each pin diagram via Tesseract.
Disabled by default (pass ``use_ocr=True`` to ``process_sheet`` to enable).

Each score region is OCR'd individually in parallel via ThreadPoolExecutor
(~1.6 s for 90 diagrams on 8 threads).  This is more reliable than the
batch-stacking approach which can't align output lines to input ROIs when
Tesseract silently skips unreadable regions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from detect import Detection

# PSM 7 = single text line — handles "1:" annotation marks correctly
# (PSM 10 = single char — misreads "1:" as "4")
_TSR_CFG = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
_WORKERS = 8


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401

        return True
    except ImportError:
        return False


def _score_roi(gray: np.ndarray, det: Detection) -> np.ndarray | None:
    """Extract and binarise the score region to the left of a detection.

    Crops a tight region just left of the pin diagram box — wide enough
    for a single handwritten digit (~0.8 box widths) but not so wide that
    annotation marks (colons, dots) from adjacent columns leak in.
    """
    w = det.width
    x0 = max(0, int(det.x_min - w * 0.9))
    x1 = max(0, int(det.x_min - w * 0.05))
    y0, y1 = max(0, det.y_min), min(gray.shape[0], det.y_max)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    scale = max(1, 48 // roi.shape[0])
    up = cv2.resize(
        roi,
        (roi.shape[1] * scale, roi.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _ocr_one(binary: np.ndarray) -> int | None:
    """Run tesseract on a single ROI.  Thread-safe."""
    import pytesseract

    try:
        text = pytesseract.image_to_string(binary, config=_TSR_CFG).strip()
    except Exception:
        return None
    # Take the first digit found (ignore trailing annotation marks).
    for ch in text:
        if ch.isdigit():
            return int(ch)
    return None


def cross_validate(
    gray: np.ndarray,
    detections: list[Detection],
    predictions: list[tuple[list[int], float]],
) -> list[int]:
    """Return detection indices where ``sum(cnn_pins) != ocr_score``.

    All tesseract calls run in parallel via ThreadPoolExecutor (subprocess-
    bound, so threads sidestep the GIL).  Returns ``[]`` when pytesseract
    is unavailable.
    """
    if not _has_tesseract():
        return []

    # Extract ROIs (fast, purely numpy/cv2).
    tasks: list[tuple[int, np.ndarray]] = []
    for idx, det in enumerate(detections):
        roi = _score_roi(gray, det)
        if roi is not None:
            tasks.append((idx, roi))
    if not tasks:
        return []

    # Launch all tesseract calls in parallel.
    det_indices = [t[0] for t in tasks]
    rois = [t[1] for t in tasks]
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        results = list(pool.map(_ocr_one, rois))

    # Cross-reference with CNN predictions.
    return [
        det_idx
        for det_idx, ocr_val in zip(det_indices, results)
        if ocr_val is not None and sum(predictions[det_idx][0]) != ocr_val
    ]
