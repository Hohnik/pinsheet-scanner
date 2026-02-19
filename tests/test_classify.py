"""Tests for the CNN classifier."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from classify import classify_pins_batch, load_classifier
from constants import NUM_PINS

WEIGHTS_PATH = Path("models/pin_classifier.pt")
FIXTURE_ALL_DOWN = Path("tests/fixtures/all_down.png")
FIXTURE_ALL_STANDING = Path("tests/fixtures/all_standing.png")

needs_weights = pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="No trained weights")


def _load(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        pytest.skip(f"Fixture not found: {path}")
    return img


@pytest.fixture(scope="module")
def classifier():
    if not WEIGHTS_PATH.exists():
        pytest.skip("No trained weights")
    return load_classifier(WEIGHTS_PATH)


class TestLoadClassifier:
    @needs_weights
    def test_returns_model_and_device(self, classifier):
        model, device = classifier
        assert isinstance(device, torch.device)
        assert not model.training

    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_classifier(tmp_path / "nonexistent.pt")


class TestClassify:
    @needs_weights
    def test_all_down(self, classifier):
        model, dev = classifier
        [(pins, conf)] = classify_pins_batch(model, [_load(FIXTURE_ALL_DOWN)], device=dev)
        assert pins == [1] * NUM_PINS and conf > 0.5

    @needs_weights
    def test_all_standing(self, classifier):
        model, dev = classifier
        [(pins, _)] = classify_pins_batch(model, [_load(FIXTURE_ALL_STANDING)], device=dev)
        assert pins == [0] * NUM_PINS

    @needs_weights
    def test_batch(self, classifier):
        model, dev = classifier
        results = classify_pins_batch(model, [_load(FIXTURE_ALL_DOWN), _load(FIXTURE_ALL_STANDING)], device=dev)
        assert len(results) == 2
        assert all(len(p) == NUM_PINS and 0 <= c <= 1 for p, c in results)

    @needs_weights
    def test_empty_batch(self, classifier):
        assert classify_pins_batch(classifier[0], [], device=classifier[1]) == []

    @needs_weights
    def test_bgr_input(self, classifier):
        model, dev = classifier
        bgr = cv2.cvtColor(_load(FIXTURE_ALL_DOWN), cv2.COLOR_GRAY2BGR)
        [(pins, _)] = classify_pins_batch(model, [bgr], device=dev)
        assert len(pins) == NUM_PINS

    @needs_weights
    @pytest.mark.parametrize("size", [(30, 25), (80, 60)])
    def test_various_sizes(self, classifier, size):
        model, dev = classifier
        [(pins, _)] = classify_pins_batch(model, [np.full(size, 200, dtype=np.uint8)], device=dev)
        assert len(pins) == NUM_PINS
