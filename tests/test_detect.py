"""Tests for detection utilities."""

import numpy as np

from detect import Detection, _cluster_by_x, crop_detections, draw_detections, sort_detections


def _det(cx=0, cy=0, w=40, h=40, conf=0.95):
    return Detection(x_center=cx, y_center=cy, width=w, height=h, confidence=conf)


class TestDetection:
    def test_bbox_properties(self):
        d = _det(cx=60, cy=45, w=100, h=50)
        assert (d.x_min, d.x_max, d.y_min, d.y_max) == (10, 110, 20, 70)


class TestClusterAndSort:
    def test_single_column(self):
        assert len(_cluster_by_x([_det(cx=100, cy=i) for i in (300, 100, 200)])) == 1

    def test_two_columns(self):
        dets = [_det(cx=400, cy=50), _det(cx=100, cy=150), _det(cx=100, cy=50), _det(cx=400, cy=150)]
        assert len(_cluster_by_x(dets)) == 2

    def test_empty(self):
        assert _cluster_by_x([]) == [] and sort_detections([]) == []

    def test_sort_reading_order(self):
        dets = [_det(cx=400, cy=50), _det(cx=100, cy=150), _det(cx=100, cy=50), _det(cx=400, cy=150)]
        ordered = sort_detections(dets)
        assert [d.column for d in ordered] == [0, 0, 1, 1]
        assert ordered[0].y_center < ordered[1].y_center

    def test_eight_columns(self):
        dets = [_det(cx=80 + c * 100, cy=50 + r * 30) for c in range(8) for r in range(15)]
        ordered = sort_detections(dets)
        assert len(ordered) == 120 and len({d.column for d in ordered}) == 8


class TestCropAndDraw:
    def test_crop_count_and_boundary(self):
        img = np.full((500, 500), 128, dtype=np.uint8)
        crops = crop_detections(img, [_det(cx=100, cy=100, w=50, h=50), _det(cx=5, cy=5, w=20, h=20)])
        assert len(crops) == 2 and all(c.size > 0 for c in crops)

    def test_draw_returns_same_shape(self):
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        assert draw_detections(img, [_det(cx=250, cy=250)]).shape == img.shape
