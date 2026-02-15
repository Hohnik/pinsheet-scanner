"""Tests for the preprocess_crop function.

preprocess_crop is the core preprocessing step that converts a raw camera/scan
crop into the normalised float32 tensor the CNN expects.  Steps:

    grayscale → resize to 64×64 → Otsu binarise → scale to [0, 1] float32

These tests verify the contract (shape, dtype, range), edge cases (tiny/large
images, already-grayscale vs BGR input), and determinism.
"""

import numpy as np
import pytest

from pinsheet_scanner.classify import preprocess_crop
from pinsheet_scanner.constants import CLASSIFIER_INPUT_SIZE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gray(h: int = 80, w: int = 60, value: int = 180) -> np.ndarray:
    """Create a uniform grayscale uint8 image."""
    return np.full((h, w), value, dtype=np.uint8)


def _bgr(h: int = 80, w: int = 60, value: int = 180) -> np.ndarray:
    """Create a uniform BGR uint8 image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _checkerboard(h: int = 80, w: int = 60) -> np.ndarray:
    """Create a checkerboard pattern with strong contrast for Otsu."""
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            if (r // 10 + c // 10) % 2 == 0:
                img[r, c] = 240
            else:
                img[r, c] = 15
    return img


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


class TestOutputContract:
    """The returned array must always have the right shape, dtype, and range."""

    def test_output_shape_matches_classifier_input_size(self):
        w, h = CLASSIFIER_INPUT_SIZE
        result = preprocess_crop(_gray())
        assert result.shape == (h, w)

    def test_output_dtype_is_float32(self):
        result = preprocess_crop(_gray())
        assert result.dtype == np.float32

    def test_output_values_in_zero_one(self):
        result = preprocess_crop(_gray())
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_only_binary_values_after_otsu(self):
        """Otsu binarisation means every pixel should be exactly 0.0 or 1.0."""
        result = preprocess_crop(_checkerboard())
        unique = set(np.unique(result))
        assert unique <= {0.0, 1.0}

    def test_custom_size(self):
        result = preprocess_crop(_gray(), size=(32, 32))
        assert result.shape == (32, 32)


# ---------------------------------------------------------------------------
# Grayscale vs BGR input
# ---------------------------------------------------------------------------


class TestInputFormats:
    """preprocess_crop must handle both grayscale and BGR inputs."""

    def test_grayscale_input(self):
        result = preprocess_crop(_gray())
        assert result.shape == (CLASSIFIER_INPUT_SIZE[1], CLASSIFIER_INPUT_SIZE[0])

    def test_bgr_input(self):
        result = preprocess_crop(_bgr())
        assert result.shape == (CLASSIFIER_INPUT_SIZE[1], CLASSIFIER_INPUT_SIZE[0])

    def test_grayscale_and_bgr_same_result_for_neutral_image(self):
        """A gray BGR image should produce the same result as its grayscale equivalent."""
        gray = _gray(value=128)
        bgr = np.stack([gray, gray, gray], axis=-1)
        result_gray = preprocess_crop(gray)
        result_bgr = preprocess_crop(bgr)
        np.testing.assert_array_equal(result_gray, result_bgr)


# ---------------------------------------------------------------------------
# Various input sizes (robustness)
# ---------------------------------------------------------------------------


class TestVariousSizes:
    """The function should handle crops of any reasonable size."""

    @pytest.mark.parametrize(
        "h,w",
        [
            (8, 8),
            (16, 12),
            (30, 25),
            (64, 64),
            (100, 80),
            (200, 300),
            (512, 512),
        ],
    )
    def test_resize_to_target(self, h: int, w: int):
        result = preprocess_crop(_gray(h=h, w=w))
        expected_h, expected_w = CLASSIFIER_INPUT_SIZE[1], CLASSIFIER_INPUT_SIZE[0]
        assert result.shape == (expected_h, expected_w)

    def test_already_correct_size(self):
        """Input already at target size should still work (no-op resize)."""
        w, h = CLASSIFIER_INPUT_SIZE
        result = preprocess_crop(_gray(h=h, w=w))
        assert result.shape == (h, w)


# ---------------------------------------------------------------------------
# Pixel-value edge cases
# ---------------------------------------------------------------------------


class TestPixelEdgeCases:
    def test_all_black_input(self):
        result = preprocess_crop(_gray(value=0))
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_all_white_input(self):
        result = preprocess_crop(_gray(value=255))
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mid_gray_input(self):
        result = preprocess_crop(_gray(value=128))
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        img = _checkerboard()
        a = preprocess_crop(img)
        b = preprocess_crop(img)
        np.testing.assert_array_equal(a, b)

    def test_does_not_modify_input(self):
        img = _checkerboard()
        original = img.copy()
        preprocess_crop(img)
        np.testing.assert_array_equal(img, original)
