"""Classical CV pipeline for reading pin state from cropped pin diagrams.

Each diagram shows 9 pins in a diamond layout:

        0
      1   2
    3   4   5
      6   7
        8

A filled/dark dot means the pin was knocked down; an empty/light dot means
it is still standing.  The functions here operate on a single cropped
diagram image (grayscale, uint8).

Strategy:
  1. Convert to grayscale if needed.
  2. Remove horizontal and vertical grid‑line remnants with morphological
     filtering (long thin kernels → open → subtract).
  3. Invert so that dark ink becomes bright signal.
  4. Gentle Gaussian blur to merge dot interiors with their outlines.
  5. Resize to a canonical square.
  6. Measure mean intensity under 9 circular masks at known pin positions.
  7. Threshold to classify each pin as knocked‑down (1) or standing (0).

This avoids Otsu binarisation, which was collapsing the contrast between
large filled dots (knocked‑down) and small ring dots (standing).
"""

from __future__ import annotations

import cv2
import numpy as np

from .constants import DEFAULT_MASK_RADIUS, DEFAULT_SIZE, PIN_POSITIONS

__all__ = ["DEFAULT_MASK_RADIUS", "DEFAULT_SIZE", "PIN_POSITIONS"]


def _remove_grid_lines(inverted: np.ndarray) -> np.ndarray:
    """Suppress horizontal and vertical grid‑line remnants.

    Expects an **inverted** image (bright ink on dark background).
    Grid lines from the original scan are the *brightest* features here
    because they were the darkest lines on the paper.

    Morphological opening with elongated kernels isolates long bright
    lines, which are then subtracted so only the compact pin dots remain.
    """
    h, w = inverted.shape[:2]

    horiz_len = max(w // 3, 5)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    horiz_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horiz_kernel)

    vert_len = max(h // 3, 5)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    vert_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vert_kernel)

    lines = cv2.add(horiz_lines, vert_lines)
    cleaned = cv2.subtract(inverted, lines)
    return cleaned


def _strip_border(gray: np.ndarray, fraction: float = 0.08) -> np.ndarray:
    """Zero‑out a thin border strip to remove any remaining cell‑edge pixels."""
    h, w = gray.shape[:2]
    bh = max(1, int(h * fraction))
    bw = max(1, int(w * fraction))
    out = gray.copy()
    out[:bh, :] = 0
    out[-bh:, :] = 0
    out[:, :bw] = 0
    out[:, -bw:] = 0
    return out


def clean_diagram(diagram: np.ndarray) -> np.ndarray:
    """Prepare a raw crop for intensity measurement.

    Returns a grayscale image where *brighter* pixels correspond to
    knocked‑down pin ink (i.e. the original dark filled dots).

    Strategy (deliberately simple):
      1. Invert — dark ink on light paper becomes bright signal on dark bg.
      2. Strip a generous border — grid‑line remnants live at the crop edges.
      3. Normalise to full [0, 255] range so threshold doesn't depend on
         scan brightness.
      4. Gaussian blur — merges ring‑dot outlines with their interior and
         smooths scanner noise.
    """
    if diagram.size == 0:
        return diagram

    if diagram.ndim == 3:
        gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY)
    else:
        gray = diagram.copy()

    inverted = cv2.bitwise_not(gray)

    inverted = _strip_border(inverted, fraction=0.15)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(inverted)

    lo, hi = float(enhanced.min()), float(enhanced.max())
    if hi - lo > 1:
        normalised = ((enhanced.astype(np.float32) - lo) / (hi - lo) * 255).astype(
            np.uint8
        )
    else:
        normalised = enhanced

    blurred = cv2.GaussianBlur(normalised, (5, 5), sigmaX=1.5)

    return blurred


def resize_diagram(
    diagram: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Resize a pin diagram to a fixed square dimension using area interpolation."""
    return cv2.resize(diagram, size, interpolation=cv2.INTER_AREA)


def build_pin_masks(
    size: tuple[int, int] = DEFAULT_SIZE,
    radius: int = DEFAULT_MASK_RADIUS,
) -> list[np.ndarray]:
    """Generate 9 circular masks, one per pin position.

    Each mask is a single‑channel uint8 image of shape (*h*, *w*) with a
    filled circle at the pin's expected location.

    *size* is ``(width, height)`` — the same convention as ``cv2.resize``.
    """
    w, h = size
    masks: list[np.ndarray] = []
    for px, py in PIN_POSITIONS:
        mask = np.zeros((h, w), dtype=np.uint8)
        cx = int(px * w)
        cy = int(py * h)
        cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
        masks.append(mask)
    return masks


def measure_pin_intensity(
    diagram: np.ndarray,
    masks: list[np.ndarray],
) -> list[float]:
    """Compute mean pixel intensity under each pin mask.

    Returns a list of 9 values in ``[0.0, 1.0]`` where **higher** means
    more bright signal (= more ink in the original = pin knocked down).
    """
    intensities: list[float] = []
    for mask in masks:
        pixel_count = cv2.countNonZero(mask)
        if pixel_count == 0:
            intensities.append(0.0)
            continue
        roi = cv2.bitwise_and(diagram, mask)
        mean_val = float(roi.sum()) / (pixel_count * 255.0)
        intensities.append(mean_val)
    return intensities


def classify_pins(
    intensities: list[float],
    threshold: float = 0.35,
) -> list[int]:
    """Threshold intensities to determine pin states.

    Returns a list of 9 binary values: ``1`` = knocked down, ``0`` = standing.
    """
    return [1 if v >= threshold else 0 for v in intensities]


def classify_pins_adaptive(
    intensities: list[float],
    low: float = 0.25,
    high: float = 0.55,
) -> list[int]:
    """Two‑pass adaptive classification.

    Pins clearly above *high* or below *low* are classified immediately.
    Ambiguous pins in between are classified relative to the median of the
    nine intensity values (above median → down, below → standing).

    This is more robust when the overall brightness of a crop varies due
    to scanning conditions.
    """
    n = len(intensities)
    result = [-1] * n

    for i, v in enumerate(intensities):
        if v >= high:
            result[i] = 1
        elif v < low:
            result[i] = 0

    if -1 in result:
        med = float(np.median(intensities))
        for i in range(n):
            if result[i] == -1:
                result[i] = 1 if intensities[i] >= med else 0

    return result


def pins_from_diagram(
    diagram: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
    radius: int = DEFAULT_MASK_RADIUS,
    threshold: float = 0.35,
    adaptive: bool = True,
    method: str = "intensity",
) -> list[int]:
    """Raw crop → list of 9 pin states.

    Args:
        diagram: Input diagram image
        size: Target size for processing
        radius: Mask radius for intensity method
        threshold: Classification threshold for intensity method
        adaptive: Use adaptive classification for intensity method
        method: Classification method - "intensity" (default) or "template"

    When method="intensity":
        When *adaptive* is True (the default), uses :func:`classify_pins_adaptive`
        which is more resilient to varying scan brightness.  Otherwise falls back
        to a simple fixed threshold.

    When method="template":
        Uses template matching against reference filled/ring dot patterns.
    """
    if method == "template":
        from .classify_template import classify_pins_template

        return classify_pins_template(diagram, size)

    cleaned = clean_diagram(diagram)
    resized = resize_diagram(cleaned, size)
    masks = build_pin_masks(size, radius)
    intensities = measure_pin_intensity(resized, masks)

    if adaptive:
        return classify_pins_adaptive(intensities)
    return classify_pins(intensities, threshold)


def pins_and_intensities_from_diagram(
    diagram: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
    radius: int = DEFAULT_MASK_RADIUS,
    threshold: float = 0.35,
    adaptive: bool = True,
) -> tuple[list[int], list[float]]:
    """Raw crop → (pins, intensities) tuple.

    Returns both the classified pin states and the intensity values used
    for classification. Useful for calculating classification confidence.

    Args:
        diagram: Raw cropped pin diagram image
        size: Target size for normalization
        radius: Mask radius for intensity measurement
        threshold: Fixed threshold (used when adaptive=False)
        adaptive: Use adaptive classification (default: True)

    Returns:
        Tuple of (pins, intensities) where:
        - pins: list of 9 binary values (1=down, 0=standing)
        - intensities: list of 9 float values [0.0, 1.0]
    """
    cleaned = clean_diagram(diagram)
    resized = resize_diagram(cleaned, size)
    masks = build_pin_masks(size, radius)
    intensities = measure_pin_intensity(resized, masks)

    if adaptive:
        pins = classify_pins_adaptive(intensities)
    else:
        pins = classify_pins(intensities, threshold)

    return pins, intensities
