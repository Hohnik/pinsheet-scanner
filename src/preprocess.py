"""Sheet pre-processing: perspective correction and contrast normalisation."""

from __future__ import annotations

import cv2
import numpy as np

RECTIFIED_WIDTH: int = 1200
RECTIFIED_HEIGHT: int = 1600


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return four corner points in (TL, TR, BR, BL) order."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def find_sheet_quad(gray: np.ndarray) -> np.ndarray | None:
    """Find four corners of the score-sheet table, or ``None``."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) >= 0.15 * img_area:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))
    return None


def rectify_sheet(
    image: np.ndarray,
    *,
    width: int = RECTIFIED_WIDTH,
    height: int = RECTIFIED_HEIGHT,
) -> np.ndarray:
    """Perspective-correct and CLAHE-normalise a raw sheet photo.

    Returns grayscale ``uint8`` of shape ``(height, width)``.
    Falls back to CLAHE-only if no quad is found.
    """
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    quad = find_sheet_quad(gray)
    if quad is None:
        return clahe.apply(gray)

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    warped = cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(quad, dst), (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return clahe.apply(warped)
