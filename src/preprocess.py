"""Sheet-level pre-processing: perspective correction and contrast normalisation.

Pipeline::

    raw photo → detect sheet quad → perspective warp → CLAHE

If no clean quadrilateral can be found (e.g. badly cropped image), the
function falls back to returning the original grayscale with CLAHE only.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Target rectified image dimensions (portrait, A4-ish aspect ratio).
RECTIFIED_WIDTH: int = 1200
RECTIFIED_HEIGHT: int = 1600


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return the four corner points in (TL, TR, BR, BL) order."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array(
        [
            pts[np.argmin(s)],   # top-left:     smallest x+y
            pts[np.argmin(d)],   # top-right:    smallest x-y
            pts[np.argmax(s)],   # bottom-right: largest  x+y
            pts[np.argmax(d)],   # bottom-left:  largest  x-y
        ],
        dtype=np.float32,
    )


def find_sheet_quad(gray: np.ndarray) -> np.ndarray | None:
    """Locate the four corners of the score-sheet table.

    Args:
        gray: Grayscale image.

    Returns:
        ``(4, 2)`` float32 array in TL/TR/BR/BL order, or ``None`` if no
        clean quadrilateral covering ≥ 15 % of the image is found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < 0.15 * img_area:
            continue
        return _order_corners(approx.reshape(4, 2).astype(np.float32))

    return None


def rectify_sheet(
    image: np.ndarray,
    *,
    width: int = RECTIFIED_WIDTH,
    height: int = RECTIFIED_HEIGHT,
) -> np.ndarray:
    """Apply perspective correction and CLAHE to a raw sheet photo.

    Args:
        image: BGR or grayscale image straight from the camera.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Grayscale ``uint8`` image of shape ``(height, width)``.
    """
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    quad = find_sheet_quad(gray)
    if quad is None:
        logger.debug("No sheet quad found — falling back to CLAHE only")
        return clahe.apply(gray)

    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        gray, M, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return clahe.apply(warped)
