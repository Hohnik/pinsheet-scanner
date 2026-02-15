"""Tests for the CNN-based pin state classifier."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pinsheet_scanner.classify import (
    classify_pins,
    classify_pins_batch,
    classify_pins_batch_with_confidence,
    classify_pins_with_confidence,
    load_classifier,
)
from pinsheet_scanner.constants import NUM_PINS, PIN_POSITIONS
from pinsheet_scanner.model import PinClassifier

WEIGHTS_PATH = Path("models/pin_classifier.pt")

needs_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(), reason="Trained weights not available"
)


def _make_synthetic_crop(pins: list[int], w: int = 50, h: int = 40) -> np.ndarray:
    """Render a clean synthetic crop matching the generator's geometry."""
    img = np.full((h, w), 200, dtype=np.uint8)
    for i, (nx, ny) in enumerate(PIN_POSITIONS):
        cx, cy = int(nx * (w - 1)), int(ny * (h - 1))
        axes = (4, 1) if pins[i] == 1 else (2, 2)
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, 40, -1)
    return img


@pytest.fixture(scope="module")
def classifier() -> tuple[PinClassifier, torch.device]:
    """Load the classifier once per test module."""
    if not WEIGHTS_PATH.exists():
        pytest.skip("Trained weights not available")
    return load_classifier(WEIGHTS_PATH)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestLoadClassifier:
    @needs_weights
    def test_returns_model_and_device(self, classifier):
        model, device = classifier
        assert isinstance(model, PinClassifier)
        assert isinstance(device, torch.device)

    @needs_weights
    def test_model_is_in_eval_mode(self, classifier):
        assert not classifier[0].training

    @needs_weights
    def test_explicit_cpu_device(self):
        model, device = load_classifier(WEIGHTS_PATH, device="cpu")
        assert device == torch.device("cpu")
        assert next(model.parameters()).device == torch.device("cpu")

    def test_missing_weights_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Classifier weights not found"):
            load_classifier(tmp_path / "nonexistent.pt")


# ---------------------------------------------------------------------------
# Single-image classification
# ---------------------------------------------------------------------------


class TestClassifyPins:
    @needs_weights
    @pytest.mark.parametrize(
        "pins",
        [
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1] * NUM_PINS,
            [0] * NUM_PINS,
        ],
        ids=["mixed", "all_down", "all_standing"],
    )
    def test_returns_nine_binary_values(self, classifier, pins):
        model, device = classifier
        result = classify_pins(model, _make_synthetic_crop(pins), device=device)
        assert len(result) == NUM_PINS
        assert all(p in (0, 1) for p in result)

    @needs_weights
    def test_all_down_correct(self, classifier):
        model, device = classifier
        assert (
            classify_pins(model, _make_synthetic_crop([1] * NUM_PINS), device=device)
            == [1] * NUM_PINS
        )

    @needs_weights
    def test_all_standing_correct(self, classifier):
        model, device = classifier
        assert (
            classify_pins(model, _make_synthetic_crop([0] * NUM_PINS), device=device)
            == [0] * NUM_PINS
        )

    @needs_weights
    def test_accepts_bgr_input(self, classifier):
        model, device = classifier
        bgr = cv2.cvtColor(
            _make_synthetic_crop([1, 0, 1, 0, 1, 0, 1, 0, 1]), cv2.COLOR_GRAY2BGR
        )
        assert len(classify_pins(model, bgr, device=device)) == NUM_PINS

    @needs_weights
    @pytest.mark.parametrize("size", [(30, 25), (50, 40), (80, 60), (120, 100)])
    def test_various_crop_sizes(self, classifier, size):
        model, device = classifier
        crop = np.full((size[1], size[0]), 200, dtype=np.uint8)
        assert len(classify_pins(model, crop, device=device)) == NUM_PINS


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


class TestClassifyPinsBatch:
    @needs_weights
    def test_batch_of_two(self, classifier):
        model, device = classifier
        results = classify_pins_batch(
            model,
            [
                _make_synthetic_crop([1] * NUM_PINS),
                _make_synthetic_crop([0] * NUM_PINS),
            ],
            device=device,
        )
        assert len(results) == 2
        assert results[0] == [1] * NUM_PINS
        assert results[1] == [0] * NUM_PINS

    @needs_weights
    def test_empty_batch(self, classifier):
        assert classify_pins_batch(classifier[0], [], device=classifier[1]) == []

    @needs_weights
    def test_single_item_batch(self, classifier):
        model, device = classifier
        results = classify_pins_batch(
            model, [_make_synthetic_crop([1, 1, 0, 0, 1, 1, 0, 0, 1])], device=device
        )
        assert len(results) == 1 and len(results[0]) == NUM_PINS


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    @needs_weights
    def test_single_returns_pins_and_confidence(self, classifier):
        model, device = classifier
        pins, conf = classify_pins_with_confidence(
            model, _make_synthetic_crop([1, 0, 1, 0, 1, 0, 1, 0, 1]), device=device
        )
        assert len(pins) == NUM_PINS
        assert isinstance(conf, float) and 0.0 <= conf <= 1.0

    @needs_weights
    def test_clear_pattern_has_reasonable_confidence(self, classifier):
        model, device = classifier
        _, conf = classify_pins_with_confidence(
            model, _make_synthetic_crop([1] * NUM_PINS), device=device
        )
        assert conf > 0.3

    @needs_weights
    def test_batch_with_confidence(self, classifier):
        model, device = classifier
        results = classify_pins_batch_with_confidence(
            model,
            [
                _make_synthetic_crop([1] * NUM_PINS),
                _make_synthetic_crop([0] * NUM_PINS),
            ],
            device=device,
        )
        assert len(results) == 2
        for pins, conf in results:
            assert len(pins) == NUM_PINS and 0.0 <= conf <= 1.0

    @needs_weights
    def test_batch_with_confidence_empty(self, classifier):
        assert (
            classify_pins_batch_with_confidence(classifier[0], [], device=classifier[1])
            == []
        )


# ---------------------------------------------------------------------------
# Real fixture
# ---------------------------------------------------------------------------


class TestRealCropFixture:
    FIXTURE = Path("tests/fixtures/sample_crop.png")

    @needs_weights
    def test_real_crop_returns_valid_result(self, classifier):
        if not self.FIXTURE.exists():
            pytest.skip("Fixture image not found")
        model, device = classifier
        crop = cv2.imread(str(self.FIXTURE))
        assert crop is not None
        result = classify_pins(model, crop, device=device)
        assert len(result) == NUM_PINS and all(p in (0, 1) for p in result)

    @needs_weights
    def test_real_crop_with_confidence(self, classifier):
        if not self.FIXTURE.exists():
            pytest.skip("Fixture image not found")
        model, device = classifier
        crop = cv2.imread(str(self.FIXTURE), cv2.IMREAD_GRAYSCALE)
        assert crop is not None
        pins, conf = classify_pins_with_confidence(model, crop, device=device)
        assert len(pins) == NUM_PINS and 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# PIN_POSITIONS sanity
# ---------------------------------------------------------------------------


class TestPinPositions:
    def test_nine_positions_in_unit_range(self):
        assert len(PIN_POSITIONS) == NUM_PINS
        for x, y in PIN_POSITIONS:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    def test_top_and_bottom_pins_centered(self):
        assert PIN_POSITIONS[0][0] == pytest.approx(0.5, abs=0.05)
        assert PIN_POSITIONS[8][0] == pytest.approx(0.5, abs=0.05)
