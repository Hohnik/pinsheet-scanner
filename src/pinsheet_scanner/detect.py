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
    column: int = -1  # assigned during sorting
    row: int = -1  # assigned during sorting

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
    """Load a trained YOLO model from disk.

    Args:
        weights_path: Path to a ``.pt`` weights file (e.g. ``best.pt``).

    Returns:
        Loaded YOLO model ready for inference.
    """
    from ultralytics import YOLO as _YOLO  # noqa: N811

    return _YOLO(str(weights_path))


def detect_pin_diagrams(
    model: object,
    image: np.ndarray,
    confidence_threshold: float = 0.25,
) -> list[Detection]:
    """Run YOLO inference on an image and return pin diagram detections.

    Args:
        model: Loaded YOLO model (returned by :func:`load_model`).
        image: BGR image as a numpy array.
        confidence_threshold: Minimum confidence to keep a detection.

    Returns:
        List of Detection objects (unsorted).
    """
    results = model(image, conf=confidence_threshold, verbose=False)  # type: ignore[operator]

    detections: list[Detection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x_c, y_c, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            detections.append(
                Detection(
                    x_center=x_c,
                    y_center=y_c,
                    width=w,
                    height=h,
                    confidence=conf,
                )
            )

    return detections


def _cluster_by_x(
    detections: list[Detection],
    gap_factor: float = 0.5,
) -> list[list[Detection]]:
    """Cluster detections into columns based on x-center proximity.

    Sorts detections by x-center, then splits into a new column whenever
    the gap between consecutive x-centers exceeds ``gap_factor`` times the
    median detection width.

    Args:
        detections: Unsorted list of detections.
        gap_factor: Multiplier on median width to determine column breaks.

    Returns:
        List of columns, where each column is a list of detections.
    """
    if not detections:
        return []

    sorted_by_x = sorted(detections, key=lambda d: d.x_center)
    median_width = float(np.median([d.width for d in sorted_by_x]))
    threshold = median_width * gap_factor

    columns: list[list[Detection]] = [[sorted_by_x[0]]]
    for det in sorted_by_x[1:]:
        prev_x = columns[-1][-1].x_center
        if det.x_center - prev_x > threshold:
            columns.append([det])
        else:
            columns[-1].append(det)

    return columns


def sort_detections(detections: list[Detection]) -> list[Detection]:
    """Sort detections into reading order: left-to-right columns, top-to-bottom rows.

    Assigns ``column`` and ``row`` indices to each detection.

    Args:
        detections: Unsorted list of detections.

    Returns:
        Detections sorted in reading order with column/row assigned.
    """
    if not detections:
        return []

    columns = _cluster_by_x(detections)

    ordered: list[Detection] = []
    for col_idx, column in enumerate(columns):
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
    """Crop detected pin diagram regions from the original image.

    Args:
        image: Original BGR or grayscale image.
        detections: Sorted list of detections.
        padding: Extra pixels to add around each bounding box.

    Returns:
        List of cropped images, one per detection, in the same order.
    """
    h, w = image.shape[:2]
    crops: list[np.ndarray] = []

    for det in detections:
        x1 = max(0, det.x_min - padding)
        y1 = max(0, det.y_min - padding)
        x2 = min(w, det.x_max + padding)
        y2 = min(h, det.y_max + padding)
        crops.append(image[y1:y2, x1:x2].copy())

    return crops


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw bounding boxes and column/row labels on an image for debugging.

    Args:
        image: BGR image (will be copied, not modified in place).
        detections: Sorted detections with column/row assigned.

    Returns:
        Annotated copy of the image.
    """
    annotated = image.copy()
    for det in detections:
        color = (0, 255, 0)
        cv2.rectangle(
            annotated,
            (det.x_min, det.y_min),
            (det.x_max, det.y_max),
            color,
            2,
        )
        label = f"c{det.column}r{det.row} {det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (det.x_min, det.y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    return annotated
