import numpy as np

from detect import (
    Detection,
    _cluster_by_x,
    crop_detections,
    draw_detections,
    sort_detections,
)


def _det(
    cx: float = 0, cy: float = 0, w: float = 40, h: float = 40, conf: float = 0.95
) -> Detection:
    """Create a Detection with sensible defaults."""
    return Detection(x_center=cx, y_center=cy, width=w, height=h, confidence=conf)


class TestDetection:
    def test_bbox_properties(self):
        d = _det(cx=60, cy=45, w=100, h=50, conf=0.9)
        assert (d.x_min, d.x_max) == (10, 110)
        assert (d.y_min, d.y_max) == (20, 70)

    def test_defaults(self):
        d = _det()
        assert d.column == -1 and d.row == -1


class TestClusterByX:
    def test_single_column(self):
        columns = _cluster_by_x([_det(cx=100, cy=i) for i in (300, 100, 200)])
        assert len(columns) == 1
        assert len(columns[0]) == 3

    def test_two_columns(self):
        dets = [
            _det(cx=400, cy=50),
            _det(cx=100, cy=150),
            _det(cx=100, cy=50),
            _det(cx=400, cy=150),
        ]
        assert len(_cluster_by_x(dets)) == 2

    def test_empty(self):
        assert _cluster_by_x([]) == []


class TestSortDetections:
    def test_single_column_top_to_bottom(self):
        ordered = sort_detections([_det(cx=100, cy=y) for y in (300, 100, 200)])
        assert [d.y_center for d in ordered] == [100, 200, 300]
        assert all(d.column == 0 for d in ordered)
        assert [d.row for d in ordered] == [0, 1, 2]

    def test_two_columns_left_to_right(self):
        dets = [
            _det(cx=400, cy=50),
            _det(cx=100, cy=150),
            _det(cx=100, cy=50),
            _det(cx=400, cy=150),
        ]
        ordered = sort_detections(dets)
        assert [d.column for d in ordered] == [0, 0, 1, 1]
        assert ordered[0].y_center < ordered[1].y_center
        assert ordered[2].y_center < ordered[3].y_center

    def test_empty(self):
        assert sort_detections([]) == []

    def test_eight_columns_like_real_sheet(self):
        """Simulate 8 columns × 15 rows."""
        dets = [
            _det(cx=80 + c * 100, cy=50 + r * 30) for c in range(8) for r in range(15)
        ]
        ordered = sort_detections(dets)

        assert len(ordered) == 120
        assert len({d.column for d in ordered}) == 8

        prev_col, prev_row = -1, -1
        for d in ordered:
            if d.column != prev_col:
                assert d.column > prev_col or prev_col == -1
                prev_col, prev_row = d.column, -1
            assert d.row > prev_row or prev_row == -1
            prev_row = d.row


class TestCropDetections:
    def test_correct_count_and_shape(self):
        img = np.full((500, 500), 128, dtype=np.uint8)
        dets = [_det(cx=100, cy=100, w=50, h=50), _det(cx=300, cy=300, w=50, h=50)]
        crops = crop_detections(img, dets)
        assert len(crops) == 2
        assert all(c.ndim == 2 for c in crops)

    def test_boundary_clamping(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        crops = crop_detections(img, [_det(cx=5, cy=5, w=20, h=20)])
        assert len(crops) == 1 and crops[0].size > 0


class TestDrawDetections:
    def test_returns_correct_shape(self):
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        assert (
            draw_detections(img, [_det(cx=250, cy=250, w=80, h=80)]).shape == img.shape
        )

    def test_empty_detections(self):
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        assert draw_detections(img, []).shape == img.shape
