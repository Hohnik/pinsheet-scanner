"""Pin diagram detection using YOLO and spatial sorting for reading order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def load_model(weights_path: Path):  # -> YOLO
    """Load a trained YOLO model from *weights_path*."""
    from ultralytics import YOLO as _YOLO  # type: ignore[attr-defined]  # noqa: N811

    return _YOLO(str(weights_path))


def detect_pin_diagrams(
    model: object,
    image: np.ndarray,
    confidence_threshold: float = 0.25,
) -> list[Detection]:
    """Run YOLO inference and return pin diagram detections (unsorted)."""
    results = model(image, conf=confidence_threshold, verbose=False)  # type: ignore[operator]

    detections: list[Detection] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x_c, y_c, w, h = box.xywh[0].tolist()
            detections.append(
                Detection(
                    x_center=x_c,
                    y_center=y_c,
                    width=w,
                    height=h,
                    confidence=float(box.conf[0]),
                )
            )

    return detections


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
