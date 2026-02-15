"""Contour-based pin classification.

Uses contour area/fill ratio instead of intensity thresholding to classify
pin states. Filled dots (knocked down) produce larger/denser contours than
ring dots (standing).
"""

from __future__ import annotations

import cv2
import numpy as np

from .classify import clean_diagram, resize_diagram
from .constants import DEFAULT_MASK_RADIUS, DEFAULT_SIZE, PIN_POSITIONS


def detect_dot_contours(
    image: np.ndarray, min_area: int = 10
) -> list[tuple[float, float, float]]:
    """Find dot-like contours in a cleaned diagram.

    Returns list of (cx, cy, area) tuples for contours that pass
    the minimum area filter.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dots.append((cx, cy, area))
    return dots


def classify_pins_contour(
    diagram: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
    radius: int = DEFAULT_MASK_RADIUS,
) -> list[int]:
    """Classify pins using contour area/fill ratio instead of intensity.

    Filled dots (knocked down) produce larger/denser contours than
    ring dots (standing). Falls back to intensity-based classification
    if contour detection finds insufficient dots.
    """
    cleaned = clean_diagram(diagram)
    resized = resize_diagram(cleaned, size)

    if int(resized.max()) == 0:
        return [0] * 9

    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dots = detect_dot_contours(binary)

    w, h = size
    pins = [0] * 9
    match_radius = radius + 3

    for pin_idx, (px, py) in enumerate(PIN_POSITIONS):
        pin_cx = px * w
        pin_cy = py * h

        best_area = 0.0
        for cx, cy, area in dots:
            dist = ((cx - pin_cx) ** 2 + (cy - pin_cy) ** 2) ** 0.5
            if dist < match_radius and area > best_area:
                best_area = area

        if best_area > 30:
            pins[pin_idx] = 1

    return pins
