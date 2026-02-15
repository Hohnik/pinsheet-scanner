"""Tests for the pipeline module."""

from pathlib import Path

import numpy as np
import pytest

from pinsheet_scanner.pipeline import DEFAULT_MODEL_PATH, SheetResult, ThrowResult


class TestDefaultModelPath:
    """Tests for DEFAULT_MODEL_PATH constant."""

    def test_is_path_object(self):
        """DEFAULT_MODEL_PATH should be a Path object."""
        assert isinstance(DEFAULT_MODEL_PATH, Path)

    def test_points_to_models_directory(self):
        """DEFAULT_MODEL_PATH should point to models/pin_diagram.pt."""
        assert DEFAULT_MODEL_PATH.name == "pin_diagram.pt"
        assert DEFAULT_MODEL_PATH.parent.name == "models"

    def test_is_relative_path(self):
        """DEFAULT_MODEL_PATH should be relative, not absolute."""
        assert not DEFAULT_MODEL_PATH.is_absolute()


class TestThrowResult:
    """Tests for ThrowResult dataclass."""

    def test_has_classification_confidence_field(self):
        """ThrowResult should have a classification_confidence field."""
        throw = ThrowResult(
            column=0,
            row=0,
            score=5,
            pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0],
            confidence=0.95,
            classification_confidence=0.75,
        )
        assert throw.classification_confidence == 0.75

    def test_classification_confidence_defaults_to_zero(self):
        """classification_confidence should default to 0.0 if not provided."""
        throw = ThrowResult(
            column=0,
            row=0,
            score=5,
            pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0],
            confidence=0.95,
        )
        assert throw.classification_confidence == 0.0


class TestThrowResultExpectedScore:
    """Tests for expected_score field on ThrowResult."""

    def test_expected_score_defaults_to_none(self):
        """expected_score should default to None."""
        throw = ThrowResult(
            column=0, row=0, score=5, pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0]
        )
        assert throw.expected_score is None

    def test_expected_score_can_be_set(self):
        """expected_score should accept an integer value."""
        throw = ThrowResult(
            column=0,
            row=0,
            score=5,
            pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0],
            expected_score=5,
        )
        assert throw.expected_score == 5


class TestSheetResultMismatches:
    """Tests for mismatches property on SheetResult."""

    def test_no_mismatches_when_scores_match(self):
        """mismatches should be empty when score equals expected_score."""
        result = SheetResult()
        result.throws.append(
            ThrowResult(
                column=0,
                row=0,
                score=5,
                pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0],
                expected_score=5,
            )
        )
        assert result.mismatches == []

    def test_mismatches_when_scores_differ(self):
        """mismatches should contain throws where score != expected_score."""
        result = SheetResult()
        result.throws.append(
            ThrowResult(
                column=0,
                row=0,
                score=5,
                pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0],
                expected_score=7,
            )
        )
        assert len(result.mismatches) == 1
        assert result.mismatches[0].score == 5
        assert result.mismatches[0].expected_score == 7

    def test_mismatches_excludes_none_expected_score(self):
        """mismatches should not include throws without expected_score."""
        result = SheetResult()
        result.throws.append(
            ThrowResult(
                column=0,
                row=0,
                score=5,
                pins_down=[1, 1, 1, 0, 1, 0, 1, 0, 0],
            )
        )
        assert result.mismatches == []


class TestProcessSheetClassificationConfidence:
    """Tests that process_sheet calculates classification confidence."""

    def test_process_sheet_sets_classification_confidence(self, tmp_path):
        """process_sheet should calculate and set classification_confidence for each throw."""
        from pinsheet_scanner.pipeline import process_sheet
        import cv2

        model_path = Path("models/pin_diagram.pt")
        if not model_path.exists():
            pytest.skip("Model file not found - run training first")

        test_image = tmp_path / "test.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_image), img)

        result = process_sheet(test_image, model_path=model_path)

        for throw in result.throws:
            assert hasattr(throw, "classification_confidence")
            assert isinstance(throw.classification_confidence, float)
            assert 0.0 <= throw.classification_confidence <= 1.0

    def test_model_not_found_raises_error(self):
        """process_sheet should raise FileNotFoundError with clear message when model doesn't exist."""
        from pinsheet_scanner.pipeline import process_sheet

        nonexistent_model = Path("models/nonexistent_model.pt")
        test_image = Path("pinsheet_example.jpeg")

        with pytest.raises(FileNotFoundError) as exc_info:
            process_sheet(test_image, model_path=nonexistent_model)

        error_msg = str(exc_info.value)
        assert (
            "Model weights not found" in error_msg
            or "Model file not found" in error_msg
        )
        assert str(nonexistent_model) in error_msg

    def test_invalid_image_raises_error(self):
        """process_sheet should handle non-existent image files gracefully."""
        from pinsheet_scanner.pipeline import process_sheet
        import cv2

        nonexistent_image = Path("nonexistent_image.jpg")

        with pytest.raises((FileNotFoundError, cv2.error)) as exc_info:
            process_sheet(nonexistent_image)

        assert exc_info.type in (FileNotFoundError, cv2.error)
