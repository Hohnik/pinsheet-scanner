"""Tests for the pipeline module."""

from pathlib import Path

import numpy as np
import pytest

from pipeline import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DETECTOR_PATH,
    SheetResult,
    ThrowResult,
)


class TestDefaultPaths:
    """Tests for default model path constants."""

    @pytest.mark.parametrize(
        "path,expected_name",
        [
            (DEFAULT_DETECTOR_PATH, "pin_diagram.pt"),
            (DEFAULT_CLASSIFIER_PATH, "pin_classifier.pt"),
        ],
    )
    def test_path_is_relative_and_in_models_dir(self, path, expected_name):
        assert isinstance(path, Path)
        assert not path.is_absolute()
        assert path.name == expected_name
        assert path.parent.name == "models"


class TestThrowResult:
    """Tests for ThrowResult dataclass."""

    def test_basic_fields(self):
        t = ThrowResult(
            column=2,
            row=5,
            score=7,
            pins_down=[1, 1, 1, 0, 1, 1, 1, 0, 0],
            confidence=0.92,
            classification_confidence=0.85,
        )
        assert (t.column, t.row, t.score) == (2, 5, 7)
        assert t.pins_down == [1, 1, 1, 0, 1, 1, 1, 0, 0]
        assert (t.confidence, t.classification_confidence) == (0.92, 0.85)

    def test_defaults(self):
        t = ThrowResult(column=0, row=0, score=0)
        assert t.pins_down == []
        assert t.confidence == 0.0
        assert t.classification_confidence == 0.0


class TestSheetResult:
    """Tests for SheetResult dataclass."""

    def test_empty_sheet(self):
        r = SheetResult()
        assert (r.throws, r.columns, r.rows_per_column, r.total_pins) == ([], 0, 0, 0)

    @pytest.mark.parametrize(
        "pins_lists,expected_total",
        [
            ([[1, 1, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 0, 0]], 8),
            ([[1] * 9], 9),
            ([[0] * 9], 0),
        ],
    )
    def test_total_pins(self, pins_lists, expected_total):
        result = SheetResult()
        for i, pins in enumerate(pins_lists):
            result.throws.append(
                ThrowResult(column=0, row=i, score=sum(pins), pins_down=pins)
            )
        assert result.total_pins == expected_total


class TestProcessSheetErrors:
    """Tests for error handling in process_sheet."""

    def test_nonexistent_detector_raises_error(self):
        from pipeline import process_sheet

        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            process_sheet(
                Path("pinsheet_example.jpeg"),
                model_path=Path("models/nonexistent_detector.pt"),
            )

    def test_nonexistent_classifier_raises_error(self, tmp_path):
        import cv2

        from pipeline import process_sheet

        if not Path("models/pin_diagram.pt").exists():
            pytest.skip("Detector model not found — run training first")

        test_image = tmp_path / "test.jpg"
        cv2.imwrite(str(test_image), np.zeros((100, 100, 3), dtype=np.uint8))

        with pytest.raises(FileNotFoundError, match="Classifier weights not found"):
            process_sheet(
                test_image,
                classifier_path=Path("models/nonexistent_classifier.pt"),
            )

    def test_nonexistent_image_raises_error(self):
        from pipeline import process_sheet

        with pytest.raises(FileNotFoundError):
            process_sheet(Path("totally_nonexistent_image.jpg"))

    @pytest.mark.integration
    def test_process_sheet_returns_valid_result(self, tmp_path):
        import cv2

        from pipeline import process_sheet

        if not (
            Path("models/pin_diagram.pt").exists()
            and Path("models/pin_classifier.pt").exists()
        ):
            pytest.skip("Model weights not found — train both models first")

        test_image = tmp_path / "test.jpg"
        cv2.imwrite(str(test_image), np.zeros((100, 100, 3), dtype=np.uint8))

        result = process_sheet(test_image)
        assert isinstance(result, SheetResult)
        for t in result.throws:
            assert isinstance(t, ThrowResult)
            assert 0.0 <= t.classification_confidence <= 1.0
            assert len(t.pins_down) == 9
            assert all(p in (0, 1) for p in t.pins_down)
