"""Pin diagram detection: classical blob analysis with YOLO fallback.

Detection strategy
------------------
1. ``detect_pin_diagrams_classical`` — model-free, works on rectified
   grayscale images from :mod:`preprocess`.  Isolates individual dot
   blobs, merges nearby dots with morphological closing, then filters
   merged blobs by size and aspect ratio.
2. ``detect_pin_diagrams_yolo`` — YOLO-based fallback for images where
   the classical approach finds fewer than ``min_classical`` diagrams.
3. ``detect_pin_diagrams`` — public entry point that tries (1) then (2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias — YOLO model objects don't expose a usable base class.
YOLOModel = Any


@dataclass
class Detection:
    """A single detected pin diagram bounding box."""

    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    column: int = -1
    row: int = -1

    @property
    def x_min(self) -> int:
        return int(self.x_center - self.width / 2)

    @property
    def y_min(self) -> int:
        return int(self.y_center - self.height / 2)

    @property
    def x_max(self) -> int:
        return int(self.x_center + self.width / 2)

    @property
    def y_max(self) -> int:
        return int(self.y_center + self.height / 2)


def load_model(weights_path: Path) -> YOLOModel:
    """Load a trained YOLO model from *weights_path*."""
    from ultralytics import YOLO as _YOLO  # type: ignore[attr-defined]  # noqa: N811

    return _YOLO(str(weights_path))


# ---------------------------------------------------------------------------
# Classical detector (A3)
# ---------------------------------------------------------------------------


def detect_pin_diagrams_classical(image: np.ndarray) -> list[Detection]:
    """Detect pin diagrams via morphological blob analysis — no model needed.

    Works best on rectified, contrast-normalised images from
    :func:`~preprocess.rectify_sheet`.

    Strategy:

    1. Otsu-binarise to isolate dark printed marks.
    2. Keep only dot-sized connected components (individual pin symbols).
    3. Morphological closing merges nearby dots into diagram-level blobs.
    4. Filter merged blobs by size and aspect ratio.

    Args:
        image: Grayscale or BGR sheet image.

    Returns:
        List of :class:`Detection` objects (``confidence`` is always 1.0).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _h, w = gray.shape

    # ── Step 1: isolate dark marks ─────────────────────────────────────────
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Step 2: keep only dot-sized blobs ─────────────────────────────────
    # Individual pin symbols are small square-ish marks (~0.3–2 % of width).
    min_side = max(2, int(w * 0.003))
    max_side = max(8, int(w * 0.020))
    min_area = min_side ** 2
    max_area = (max_side * 2) ** 2

    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(binary)
    dot_mask = np.zeros_like(binary)
    for i in range(1, n_cc):
        bw = cc_stats[i, cv2.CC_STAT_WIDTH]
        bh = cc_stats[i, cv2.CC_STAT_HEIGHT]
        area = cc_stats[i, cv2.CC_STAT_AREA]
        if (
            min_side <= bw <= max_side
            and min_side <= bh <= max_side
            and min_area <= area <= max_area
            and 0.25 <= bw / max(bh, 1) <= 4.0
        ):
            dot_mask[cc_labels == i] = 255

    # ── Step 3: close nearby dots into diagram-level blobs ────────────────
    # Kernel large enough to bridge within-diagram gaps, small enough not
    # to merge adjacent diagram columns (~2.5 % of image width).
    close_px = max(8, int(w * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    merged = cv2.morphologyEx(dot_mask, cv2.MORPH_CLOSE, kernel)

    # ── Step 4: filter diagram-level blobs by size and shape ──────────────
    min_diag = max(10, int(w * 0.025))
    max_diag = max(50, int(w * 0.12))

    n_diag, _, d_stats, d_centroids = cv2.connectedComponentsWithStats(merged)
    detections: list[Detection] = []
    for i in range(1, n_diag):
        dw = d_stats[i, cv2.CC_STAT_WIDTH]
        dh = d_stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = d_centroids[i]
        if not (min_diag <= dw <= max_diag and min_diag <= dh <= max_diag):
            continue
        if not (0.4 <= dw / max(dh, 1) <= 2.5):
            continue
        detections.append(
            Detection(
                x_center=float(cx),
                y_center=float(cy),
                width=float(dw),
                height=float(dh),
                confidence=1.0,
            )
        )

    return detections


# ---------------------------------------------------------------------------
# YOLO fallback
# ---------------------------------------------------------------------------


def detect_pin_diagrams_yolo(
    model: YOLOModel,
    image: np.ndarray,
    confidence_threshold: float = 0.25,
) -> list[Detection]:
    """Run YOLO inference and return unsorted pin diagram detections."""
    # YOLO expects a 3-channel BGR image.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results = model(image, conf=confidence_threshold, verbose=False)
    detections: list[Detection] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x_c, y_c, bw, bh = box.xywh[0].tolist()
            detections.append(
                Detection(
                    x_center=x_c,
                    y_center=y_c,
                    width=bw,
                    height=bh,
                    confidence=float(box.conf[0]),
                )
            )
    return detections


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_pin_diagrams(
    model: YOLOModel | None,
    image: np.ndarray,
    confidence_threshold: float = 0.25,
    *,
    min_classical: int = 6,
) -> list[Detection]:
    """Detect pin diagrams — classical first, YOLO fallback.

    Args:
        model: YOLO model instance, or ``None`` to force classical only.
        image: BGR or grayscale sheet image (ideally rectified).
        confidence_threshold: YOLO confidence threshold (fallback only).
        min_classical: Accept classical result if it finds at least this
            many diagrams; otherwise fall back to YOLO.

    Returns:
        List of unsorted :class:`Detection` objects.
    """
    detections = detect_pin_diagrams_classical(image)
    if len(detections) >= min_classical or model is None:
        logger.debug(
            "Classical detector found %d diagrams (threshold=%d)",
            len(detections), min_classical,
        )
        return detections
    logger.info(
        "Classical detector found only %d diagrams (< %d), falling back to YOLO",
        len(detections), min_classical,
    )
    return detect_pin_diagrams_yolo(model, image, confidence_threshold)


# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------


def _cluster_by_x(
    detections: list[Detection],
    gap_factor: float = 0.5,
) -> list[list[Detection]]:
    """Cluster detections into columns based on x-center proximity."""
    if not detections:
        return []

    sorted_by_x = sorted(detections, key=lambda d: d.x_center)
    threshold = float(np.median([d.width for d in sorted_by_x])) * gap_factor

    columns: list[list[Detection]] = [[sorted_by_x[0]]]
    for det in sorted_by_x[1:]:
        if det.x_center - columns[-1][-1].x_center > threshold:
            columns.append([det])
        else:
            columns[-1].append(det)
    return columns


def sort_detections(detections: list[Detection]) -> list[Detection]:
    """Sort detections into reading order (left→right columns, top→bottom rows).

    Assigns ``column`` and ``row`` indices to each detection.
    """
    if not detections:
        return []

    ordered: list[Detection] = []
    for col_idx, column in enumerate(_cluster_by_x(detections)):
        column.sort(key=lambda d: d.y_center)
        for row_idx, det in enumerate(column):
            det.column = col_idx
            det.row = row_idx
            ordered.append(det)
    return ordered


def crop_detections(
    image: np.ndarray,
    detections: list[Detection],
    padding: int = 2,
) -> list[np.ndarray]:
    """Crop detected pin diagram regions from *image*."""
    h, w = image.shape[:2]
    return [
        image[
            max(0, d.y_min - padding) : min(h, d.y_max + padding),
            max(0, d.x_min - padding) : min(w, d.x_max + padding),
        ].copy()
        for d in detections
    ]


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of *image* for debugging."""
    annotated = image.copy()
    for det in detections:
        color = (0, 255, 0)
        cv2.rectangle(
            annotated, (det.x_min, det.y_min), (det.x_max, det.y_max), color, 2
        )
        cv2.putText(
            annotated,
            f"c{det.column}r{det.row} {det.confidence:.2f}",
            (det.x_min, det.y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    return annotated
