import numpy as np

from pinsheet_scanner.detect import (
    Detection,
    _cluster_by_x,
    crop_detections,
    draw_detections,
    sort_detections,
)


class TestDetection:
    def test_x_center_stored(self):
        d = Detection(
            x_center=60.0, y_center=45.0, width=100, height=50, confidence=0.9
        )
        assert d.x_center == 60.0

    def test_y_center_stored(self):
        d = Detection(
            x_center=60.0, y_center=45.0, width=100, height=50, confidence=0.9
        )
        assert d.y_center == 45.0

    def test_x_min(self):
        d = Detection(
            x_center=60.0, y_center=45.0, width=100, height=50, confidence=0.9
        )
        assert d.x_min == 10

    def test_x_max(self):
        d = Detection(
            x_center=60.0, y_center=45.0, width=100, height=50, confidence=0.9
        )
        assert d.x_max == 110

    def test_y_min(self):
        d = Detection(
            x_center=60.0, y_center=45.0, width=100, height=50, confidence=0.9
        )
        assert d.y_min == 20

    def test_y_max(self):
        d = Detection(
            x_center=60.0, y_center=45.0, width=100, height=50, confidence=0.9
        )
        assert d.y_max == 70

    def test_default_column_and_row(self):
        d = Detection(x_center=0, y_center=0, width=10, height=10, confidence=0.5)
        assert d.column == -1
        assert d.row == -1


class TestClusterByX:
    def _make(self, cx: float, cy: float) -> Detection:
        """Helper to create a detection with a given center."""
        w, h = 40, 40
        return Detection(x_center=cx, y_center=cy, width=w, height=h, confidence=0.95)

    def test_single_column(self):
        detections = [
            self._make(100, 300),
            self._make(100, 100),
            self._make(100, 200),
        ]
        columns = _cluster_by_x(detections)
        assert len(columns) == 1
        assert len(columns[0]) == 3

    def test_two_columns(self):
        detections = [
            self._make(400, 50),
            self._make(100, 150),
            self._make(100, 50),
            self._make(400, 150),
        ]
        columns = _cluster_by_x(detections)
        assert len(columns) == 2

    def test_empty_input(self):
        columns = _cluster_by_x([])
        assert columns == []


class TestSortDetections:
    def _make(self, cx: float, cy: float) -> Detection:
        """Helper to create a detection with a given center."""
        w, h = 40, 40
        return Detection(x_center=cx, y_center=cy, width=w, height=h, confidence=0.95)

    def test_single_column_sorted_top_to_bottom(self):
        detections = [
            self._make(100, 300),
            self._make(100, 100),
            self._make(100, 200),
        ]
        ordered = sort_detections(detections)
        assert len(ordered) == 3
        ys = [d.y_center for d in ordered]
        assert ys == sorted(ys)
        for d in ordered:
            assert d.column == 0

    def test_two_columns_left_to_right(self):
        detections = [
            self._make(400, 50),
            self._make(100, 150),
            self._make(100, 50),
            self._make(400, 150),
        ]
        ordered = sort_detections(detections)
        assert len(ordered) == 4
        assert ordered[0].column == 0
        assert ordered[1].column == 0
        assert ordered[2].column == 1
        assert ordered[3].column == 1
        assert ordered[0].y_center < ordered[1].y_center
        assert ordered[2].y_center < ordered[3].y_center

    def test_empty_input(self):
        ordered = sort_detections([])
        assert ordered == []

    def test_row_indices_assigned(self):
        detections = [
            self._make(100, 300),
            self._make(100, 100),
            self._make(100, 200),
        ]
        ordered = sort_detections(detections)
        rows = [d.row for d in ordered]
        assert rows == [0, 1, 2]

    def test_eight_columns_like_real_sheet(self):
        """Simulate 8 columns (4 Bahns x 2) with 15 rows each."""
        detections = []
        for col_idx in range(8):
            cx = 80 + col_idx * 100
            for row_idx in range(15):
                cy = 50 + row_idx * 30
                detections.append(self._make(cx, cy))

        ordered = sort_detections(detections)
        assert len(ordered) == 120

        col_ids = set(d.column for d in ordered)
        assert len(col_ids) == 8

        prev_col = -1
        prev_row = -1
        for d in ordered:
            if d.column != prev_col:
                assert d.column > prev_col or prev_col == -1
                prev_col = d.column
                prev_row = -1
            assert d.row > prev_row or prev_row == -1
            prev_row = d.row


class TestCropDetections:
    def test_returns_correct_number_of_crops(self):
        img = np.full((500, 500), 128, dtype=np.uint8)
        dets = [
            Detection(x_center=100, y_center=100, width=50, height=50, confidence=0.9),
            Detection(x_center=300, y_center=300, width=50, height=50, confidence=0.9),
        ]
        crops = crop_detections(img, dets)
        assert len(crops) == 2

    def test_crop_is_2d_array(self):
        img = np.full((500, 500), 128, dtype=np.uint8)
        dets = [
            Detection(x_center=250, y_center=250, width=80, height=80, confidence=0.9)
        ]
        crops = crop_detections(img, dets)
        assert crops[0].ndim == 2

    def test_boundary_clamping(self):
        """Detections near image edge should not crash."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        dets = [Detection(x_center=5, y_center=5, width=20, height=20, confidence=0.9)]
        crops = crop_detections(img, dets)
        assert len(crops) == 1
        assert crops[0].size > 0


class TestDrawDetections:
    def test_returns_image_of_correct_shape(self):
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        dets = [
            Detection(x_center=250, y_center=250, width=80, height=80, confidence=0.9)
        ]
        annotated = draw_detections(img, dets)
        assert annotated.shape == img.shape

    def test_no_crash_on_empty_detections(self):
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        annotated = draw_detections(img, [])
        assert annotated.shape == img.shape
