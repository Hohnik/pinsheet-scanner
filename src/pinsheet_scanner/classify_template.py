"""Template matching based pin classification."""

import cv2
import numpy as np


def create_templates(size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Create reference templates for filled (down) and ring (standing) dots.

    Args:
        size: Template size in pixels (default: 16x16)

    Returns:
        Tuple of (filled_template, ring_template) as grayscale images
    """
    filled = np.ones((size, size), dtype=np.uint8) * 255
    center = size // 2
    radius = size // 3
    cv2.circle(filled, (center, center), radius, 0, -1)

    ring = np.ones((size, size), dtype=np.uint8) * 255
    cv2.circle(ring, (center, center), radius, 0, 2)

    return filled, ring


def classify_pins_template(
    diagram: np.ndarray,
    size: tuple[int, int] | None = None,
    template_size: int = 16,
) -> list[int]:
    """Classify pin states using template matching.

    Args:
        diagram: Input diagram image (grayscale or color)
        size: Expected diagram size (default: 64x64 from classify.DEFAULT_SIZE)
        template_size: Size of templates in pixels (default: 16)

    Returns:
        List of 9 pin states (1=down, 0=standing)
    """
    from .constants import DEFAULT_SIZE, PIN_POSITIONS

    if size is None:
        size = DEFAULT_SIZE

    if diagram.ndim == 3:
        gray = cv2.cvtColor(diagram, cv2.COLOR_BGR2GRAY)
    else:
        gray = diagram.copy()

    if gray.shape[:2] != (size[1], size[0]):
        gray = cv2.resize(gray, size)

    filled_template, ring_template = create_templates(template_size)

    results = []

    for norm_x, norm_y in PIN_POSITIONS:
        px = int(norm_x * (size[0] - 1))
        py = int(norm_y * (size[1] - 1))

        half = template_size // 2
        x1 = max(0, px - half)
        y1 = max(0, py - half)
        x2 = min(size[0], px + half)
        y2 = min(size[1], py + half)

        region = gray[y1:y2, x1:x2]

        if region.shape[0] < template_size // 2 or region.shape[1] < template_size // 2:
            results.append(1)
            continue

        region_resized = cv2.resize(region, (template_size, template_size))

        result_filled = cv2.matchTemplate(
            region_resized, filled_template, cv2.TM_CCOEFF_NORMED
        )
        result_ring = cv2.matchTemplate(
            region_resized, ring_template, cv2.TM_CCOEFF_NORMED
        )

        score_filled = np.max(result_filled)
        score_ring = np.max(result_ring)

        if score_filled > score_ring:
            results.append(1)
        else:
            results.append(0)

    return results
