"""Tests for the pipeline module."""

from pathlib import Path

import numpy as np
import pytest

from pipeline import SheetResult, ThrowResult


class TestDataclasses:
    def test_empty_sheet(self):
        r = SheetResult()
        assert r.throws == [] and r.total_pins == 0

    def test_total_pins(self):
        r = SheetResult()
        r.throws.append(ThrowResult(column=0, row=0, score=3, pins_down=[1, 1, 1, 0, 0, 0, 0, 0, 0]))
        r.throws.append(ThrowResult(column=0, row=1, score=9, pins_down=[1] * 9))
        assert r.total_pins == 12

    def test_throw_defaults(self):
        t = ThrowResult(column=0, row=0, score=0)
        assert t.pins_down == [] and t.confidence == 0.0


class TestProcessSheetErrors:
    def test_missing_image_raises(self):
        from pipeline import process_sheet

        with pytest.raises(FileNotFoundError):
            process_sheet(Path("nonexistent.jpg"))

    def test_missing_detector_is_ok(self, tmp_path):
        """Classical detection runs when YOLO weights are missing."""
        import cv2

        from pipeline import process_sheet

        img = tmp_path / "test.jpg"
        cv2.imwrite(str(img), np.zeros((100, 100, 3), dtype=np.uint8))
        result = process_sheet(img, model_path=Path("nonexistent.pt"),
                               classifier_path=Path("nonexistent.pt"))
        assert result.throws == []

    def test_missing_classifier_raises(self):
        from classify import load_classifier

        with pytest.raises(FileNotFoundError):
            load_classifier(Path("nonexistent.pt"))
