"""OCR cross-validation of CNN pin-count predictions.

Reads the printed score digit adjacent to each pin diagram via Tesseract.
Disabled by default (pass ``use_ocr=True`` to ``process_sheet`` to enable).

Strategy: stack all score regions into one image → one tesseract call (~200 ms).
Fall back to parallel per-ROI calls for any rows that produce ambiguous output.
Total overhead: ~200–400 ms/sheet instead of ~6 s serial.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from detect import Detection

_TSR_CFG_BATCH = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
_TSR_CFG_SINGLE = "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
_WORKERS = 8


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
    y0, y1 = max(0, det.y_min), min(gray.shape[0], det.y_max)
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


def _parse_digit(text: str) -> int | None:
    """Extract a single digit 0-9 from OCR text, or None."""
    digits = "".join(c for c in text if c.isdigit())
    if len(digits) == 1:
        return int(digits)
    return None


def _ocr_one(binary: np.ndarray) -> int | None:
    """Run tesseract on a single ROI.  Thread-safe."""
    import pytesseract
    try:
        text = pytesseract.image_to_string(binary, config=_TSR_CFG_SINGLE).strip()
    except Exception:
        return None
    return _parse_digit(text)


def _batch_ocr(rois: list[np.ndarray]) -> list[int | None]:
    """Stack ROIs into one image, run tesseract once (~200 ms).

    Returns a list aligned with *rois*.  Entries are ``None`` where
    tesseract produced no digit or an ambiguous result.
    """
    import pytesseract

    target_h = max(r.shape[0] for r in rois)
    max_w = max(r.shape[1] for r in rois)
    sep_h = target_h  # full-height separator to prevent merging

    strips: list[np.ndarray] = []
    for roi in rois:
        pad = target_h - roi.shape[0]
        top, bot = pad // 2, pad - pad // 2
        padded = cv2.copyMakeBorder(roi, top, bot, 0, max_w - roi.shape[1],
                                    cv2.BORDER_CONSTANT, value=255)
        strips.append(padded)
        strips.append(np.full((sep_h, max_w), 255, dtype=np.uint8))

    try:
        raw = pytesseract.image_to_string(np.vstack(strips), config=_TSR_CFG_BATCH)
    except Exception:
        return [None] * len(rois)

    # Parse non-blank lines.
    parsed: list[int | None] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        parsed.append(_parse_digit(s))

    # Pad / truncate to match input length (tesseract may skip blank regions).
    while len(parsed) < len(rois):
        parsed.append(None)
    return parsed[:len(rois)]


def cross_validate(
    gray: np.ndarray,
    detections: list[Detection],
    predictions: list[tuple[list[int], float]],
) -> list[int]:
    """Return indices where ``sum(cnn_pins) != ocr_score``.

    Fast path: one tesseract call for all ROIs (~200 ms).
    Fallback: parallel per-ROI calls for ambiguous results (~65 ms each, 8 threads).
    """
    if not _has_tesseract():
        return []

    # Extract ROIs.
    tasks: list[tuple[int, np.ndarray]] = []
    for idx, det in enumerate(detections):
        roi = _score_roi(gray, det)
        if roi is not None:
            tasks.append((idx, roi))
    if not tasks:
        return []

    det_indices = [t[0] for t in tasks]
    rois = [t[1] for t in tasks]

    # Fast batch pass.
    results = _batch_ocr(rois)

    # Find ambiguous entries (None) and retry them in parallel.
    retry = [(i, rois[i]) for i, v in enumerate(results) if v is None]
    if retry:
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            retried = list(pool.map(lambda t: _ocr_one(t[1]), retry))
        for (slot, _), val in zip(retry, retried):
            results[slot] = val

    # Cross-reference.
    return [
        det_indices[i]
        for i, ocr_val in enumerate(results)
        if ocr_val is not None and sum(predictions[det_indices[i]][0]) != ocr_val
    ]
