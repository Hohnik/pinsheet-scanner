"""OCR-based score extraction for ground truth validation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pinsheet_scanner.detect import Detection


def extract_row_scores(
    image: np.ndarray, detections: list[Detection]
) -> list[int | None]:
    """Extract printed score digits from regions to the left of pin diagrams.

    Args:
        image: Grayscale image of the score sheet.
        detections: List of Detection objects with bounding box properties.

    Returns:
        List of extracted scores (0-9) or None for unrecognizable text.
        One entry per detection.
    """
    scores = []

    for det in detections:
        score_width = min(int(det.width) // 2, 80)
        score_height = int(det.height)

        x_start = max(0, det.x_min - score_width)
        x_end = det.x_min
        y_start = max(0, det.y_min)
        y_end = min(image.shape[0], det.y_min + score_height)

        score_region = image[y_start:y_end, x_start:x_end]

        if score_region.size == 0:
            scores.append(None)
            continue

        _, binary = cv2.threshold(
            score_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        custom_config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"

        try:
            raw_text = pytesseract.image_to_string(binary, config=custom_config)
            text = raw_text if isinstance(raw_text, str) else str(raw_text)
            text = text.strip()

            digit_match = re.search(r"\d", text)
            if digit_match:
                scores.append(int(digit_match.group()))
            else:
                scores.append(None)
        except (
            pytesseract.TesseractError,
            pytesseract.TesseractNotFoundError,
            ValueError,
        ):
            scores.append(None)

    return scores
