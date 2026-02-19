"""Tests for the CNN-based pin state classifier."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from classify import (
    classify_pins_batch_with_confidence,
    load_classifier,
)
from constants import NUM_PINS
from model import PinClassifier

WEIGHTS_PATH = Path("models/pin_classifier.pt")
FIXTURE_ALL_DOWN = Path("tests/fixtures/all_down.png")
FIXTURE_ALL_STANDING = Path("tests/fixtures/all_standing.png")

needs_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(), reason="Trained weights not available"
)


def _load_fixture(path: Path) -> np.ndarray:
    """Load a real crop fixture as grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return img


def _classify_one(
    model: PinClassifier,
    crop: np.ndarray,
    *,
    device: torch.device | None = None,
) -> tuple[list[int], float]:
    """Convenience: classify a single crop via the batch API."""
    return classify_pins_batch_with_confidence(model, [crop], device=device)[0]


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
# Single-image classification — shape / contract checks
# ---------------------------------------------------------------------------


class TestClassifySingle:
    @needs_weights
    def test_returns_nine_binary_values(self, classifier):
        model, device = classifier
        crop = _load_fixture(FIXTURE_ALL_DOWN)
        pins, conf = _classify_one(model, crop, device=device)
        assert len(pins) == NUM_PINS
        assert all(p in (0, 1) for p in pins)

    @needs_weights
    def test_all_down_correct(self, classifier):
        model, device = classifier
        crop = _load_fixture(FIXTURE_ALL_DOWN)
        pins, _ = _classify_one(model, crop, device=device)
        assert pins == [1] * NUM_PINS

    @needs_weights
    def test_all_standing_correct(self, classifier):
        model, device = classifier
        crop = _load_fixture(FIXTURE_ALL_STANDING)
        pins, _ = _classify_one(model, crop, device=device)
        assert pins == [0] * NUM_PINS

    @needs_weights
    def test_accepts_bgr_input(self, classifier):
        model, device = classifier
        gray = _load_fixture(FIXTURE_ALL_DOWN)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        pins, _ = _classify_one(model, bgr, device=device)
        assert len(pins) == NUM_PINS
        assert all(p in (0, 1) for p in pins)

    @needs_weights
    @pytest.mark.parametrize("size", [(30, 25), (50, 40), (80, 60), (120, 100)])
    def test_various_crop_sizes(self, classifier, size):
        model, device = classifier
        crop = np.full((size[1], size[0]), 200, dtype=np.uint8)
        pins, _ = _classify_one(model, crop, device=device)
        assert len(pins) == NUM_PINS
        assert all(p in (0, 1) for p in pins)


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


class TestClassifyBatch:
    @needs_weights
    def test_batch_of_two(self, classifier):
        model, device = classifier
        results = classify_pins_batch_with_confidence(
            model,
            [
                _load_fixture(FIXTURE_ALL_DOWN),
                _load_fixture(FIXTURE_ALL_STANDING),
            ],
            device=device,
        )
        assert len(results) == 2
        assert results[0][0] == [1] * NUM_PINS
        assert results[1][0] == [0] * NUM_PINS

    @needs_weights
    def test_empty_batch(self, classifier):
        assert (
            classify_pins_batch_with_confidence(classifier[0], [], device=classifier[1])
            == []
        )

    @needs_weights
    def test_single_item_batch(self, classifier):
        model, device = classifier
        results = classify_pins_batch_with_confidence(
            model, [_load_fixture(FIXTURE_ALL_DOWN)], device=device
        )
        assert len(results) == 1
        pins, conf = results[0]
        assert len(pins) == NUM_PINS


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    @needs_weights
    def test_returns_float_in_range(self, classifier):
        model, device = classifier
        _, conf = _classify_one(model, _load_fixture(FIXTURE_ALL_DOWN), device=device)
        assert isinstance(conf, float) and 0.0 <= conf <= 1.0

    @needs_weights
    def test_clear_pattern_has_high_confidence(self, classifier):
        model, device = classifier
        _, conf = _classify_one(model, _load_fixture(FIXTURE_ALL_DOWN), device=device)
        assert conf > 0.5

    @needs_weights
    def test_batch_confidence(self, classifier):
        model, device = classifier
        results = classify_pins_batch_with_confidence(
            model,
            [
                _load_fixture(FIXTURE_ALL_DOWN),
                _load_fixture(FIXTURE_ALL_STANDING),
            ],
            device=device,
        )
        assert len(results) == 2
        for pins, conf in results:
            assert len(pins) == NUM_PINS and 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Real fixture (sample_crop)
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
        pins, _ = _classify_one(model, crop, device=device)
        assert len(pins) == NUM_PINS and all(p in (0, 1) for p in pins)

    @needs_weights
    def test_real_crop_with_confidence(self, classifier):
        if not self.FIXTURE.exists():
            pytest.skip("Fixture image not found")
        model, device = classifier
        crop = cv2.imread(str(self.FIXTURE), cv2.IMREAD_GRAYSCALE)
        assert crop is not None
        pins, conf = _classify_one(model, crop, device=device)
        assert len(pins) == NUM_PINS and 0.0 <= conf <= 1.0
