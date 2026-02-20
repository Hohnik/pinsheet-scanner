"""OCR cross-validation of CNN pin-count predictions.

Reads the printed score digit adjacent to each pin diagram via Tesseract.
Disabled by default (pass ``use_ocr=True`` to ``process_sheet`` to enable).

Each score region is OCR'd individually in parallel via ThreadPoolExecutor
(~1.6 s for 90 diagrams on 8 threads).  Only digits recognised with
confidence ≥ 60 are used for cross-validation — this filters out tesseract
misreads on faint or ambiguous glyphs while keeping genuine mismatches.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from detect import Detection

# PSM 7 = single text line (handles "1:" annotation marks).
# Digit-only whitelist so tesseract doesn't emit letters/punctuation.
_TSR_CFG = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
_TARGET_H = 96  # upscale small ROIs so printed digits are clearly resolved
_MIN_CONF = 60  # ignore OCR reads below this confidence (0-100)
_WORKERS = 8


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401

        return True
    except ImportError:
        return False


def _score_roi(gray: np.ndarray, det: Detection) -> np.ndarray | None:
    """Extract the score region just left of a detection box."""
    w = det.width
    x0 = max(0, int(det.x_min - w * 0.9))
    x1 = max(0, int(det.x_min - w * 0.05))
    y0, y1 = max(0, det.y_min), min(gray.shape[0], det.y_max)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = gray[y0:y1, x0:x1]
    return roi if roi.size > 0 else None


def _ocr_one(roi: np.ndarray) -> int | None:
    """Run tesseract on a single score-region ROI.  Thread-safe.

    Uses ``image_to_data`` to get per-character confidence.  Only returns
    a digit if tesseract's confidence is ≥ ``_MIN_CONF``.

    Strategy: upscale to ~96 px tall, try the grayscale image first
    (preserves subtle stroke detail for printed digits).  If that yields
    nothing, retry on a binarised version (recovers low-contrast ROIs).
    """
    import pytesseract

    scale = max(1, _TARGET_H // roi.shape[0])
    up = cv2.resize(
        roi,
        (roi.shape[1] * scale, roi.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )

    for img in [up, None]:  # None = trigger binarised fallback
        if img is None:
            _, img = cv2.threshold(
                up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        try:
            data = pytesseract.image_to_data(
                img,
                config=_TSR_CFG,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        for txt, conf in zip(data["text"], data["conf"]):
            txt = txt.strip()
            if txt and txt[0].isdigit() and int(conf) >= _MIN_CONF:
                return int(txt[0])
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

    tasks: list[tuple[int, np.ndarray]] = []
    for idx, det in enumerate(detections):
        roi = _score_roi(gray, det)
        if roi is not None:
            tasks.append((idx, roi))
    if not tasks:
        return []

    det_indices = [t[0] for t in tasks]
    rois = [t[1] for t in tasks]
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        results = list(pool.map(_ocr_one, rois))

    return [
        det_idx
        for det_idx, ocr_val in zip(det_indices, results)
        if ocr_val is not None and sum(predictions[det_idx][0]) != ocr_val
    ]
