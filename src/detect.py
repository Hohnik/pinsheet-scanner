"""Pin diagram detection: classical blob analysis with YOLO fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

YOLOModel = Any


@dataclass
class Detection:
    """A detected pin diagram bounding box."""

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
    """Load a trained YOLO model."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    return YOLO(str(weights_path))


def detect_pin_diagrams_classical(image: np.ndarray) -> list[Detection]:
    """Detect pin diagrams via morphological blob analysis — no model needed.

    Strategy: Otsu binarise → keep dot-sized blobs → morphological close
    to merge nearby dots → filter merged blobs by size and aspect ratio.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep only dot-sized blobs (individual pin symbols).
    min_side, max_side = max(2, int(w * 0.003)), max(8, int(w * 0.020))
    min_area, max_area = min_side ** 2, (max_side * 2) ** 2

    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(binary)
    dot_mask = np.zeros_like(binary)
    for i in range(1, n_cc):
        bw = cc_stats[i, cv2.CC_STAT_WIDTH]
        bh = cc_stats[i, cv2.CC_STAT_HEIGHT]
        area = cc_stats[i, cv2.CC_STAT_AREA]
        if (min_side <= bw <= max_side and min_side <= bh <= max_side
                and min_area <= area <= max_area and 0.25 <= bw / max(bh, 1) <= 4.0):
            dot_mask[cc_labels == i] = 255

    # Close nearby dots into diagram-level blobs.
    close_px = max(8, int(w * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    merged = cv2.morphologyEx(dot_mask, cv2.MORPH_CLOSE, kernel)

    # Filter diagram-level blobs by size and shape.
    min_diag, max_diag = max(10, int(w * 0.025)), max(50, int(w * 0.12))
    n_diag, _, d_stats, d_centroids = cv2.connectedComponentsWithStats(merged)
    detections: list[Detection] = []
    for i in range(1, n_diag):
        dw, dh = d_stats[i, cv2.CC_STAT_WIDTH], d_stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = d_centroids[i]
        if (min_diag <= dw <= max_diag and min_diag <= dh <= max_diag
                and 0.4 <= dw / max(dh, 1) <= 2.5):
            detections.append(Detection(float(cx), float(cy), float(dw), float(dh), 1.0))
    return detections


def detect_pin_diagrams_yolo(
    model: YOLOModel, image: np.ndarray, confidence_threshold: float = 0.25,
) -> list[Detection]:
    """Run YOLO inference and return unsorted detections."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detections: list[Detection] = []
    for result in model(image, conf=confidence_threshold, verbose=False):
        if result.boxes is None:
            continue
        for box in result.boxes:
            x_c, y_c, bw, bh = box.xywh[0].tolist()
            detections.append(Detection(x_c, y_c, bw, bh, float(box.conf[0])))
    return detections


def detect_pin_diagrams(
    model: YOLOModel | None,
    image: np.ndarray,
    confidence_threshold: float = 0.25,
    *,
    min_classical: int = 6,
) -> list[Detection]:
    """Classical detection first; YOLO fallback if fewer than *min_classical* found."""
    detections = detect_pin_diagrams_classical(image)
    if len(detections) >= min_classical or model is None:
        return detections
    return detect_pin_diagrams_yolo(model, image, confidence_threshold)


def _cluster_by_x(detections: list[Detection], gap_factor: float = 0.5) -> list[list[Detection]]:
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
    """Sort into reading order (left→right columns, top→bottom rows)."""
    if not detections:
        return []
    ordered: list[Detection] = []
    for col_idx, column in enumerate(_cluster_by_x(detections)):
        column.sort(key=lambda d: d.y_center)
        for row_idx, det in enumerate(column):
            det.column, det.row = col_idx, row_idx
            ordered.append(det)
    return ordered


def crop_detections(image: np.ndarray, detections: list[Detection], padding: int = 2) -> list[np.ndarray]:
    """Crop detected regions from *image*."""
    h, w = image.shape[:2]
    return [
        image[max(0, d.y_min - padding):min(h, d.y_max + padding),
              max(0, d.x_min - padding):min(w, d.x_max + padding)].copy()
        for d in detections
    ]


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes on a copy of *image* for debugging."""
    out = image.copy()
    for det in detections:
        cv2.rectangle(out, (det.x_min, det.y_min), (det.x_max, det.y_max), (0, 255, 0), 2)
        cv2.putText(out, f"c{det.column}r{det.row} {det.confidence:.2f}",
                    (det.x_min, det.y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out
