"""Tests for preprocess_crop."""

import numpy as np
import pytest

from classify import preprocess_crop


def _gray(h=80, w=60, v=180):
    return np.full((h, w), v, dtype=np.uint8)


def _checkerboard(h=80, w=60):
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            img[r, c] = 240 if (r // 10 + c // 10) % 2 == 0 else 15
    return img


class TestOutputContract:
    def test_shape_dtype_range(self):
        result = preprocess_crop(_gray())
        assert result.shape == (64, 64) and result.dtype == np.float32
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_binary_after_otsu(self):
        assert set(np.unique(preprocess_crop(_checkerboard()))) <= {0.0, 1.0}


class TestInputFormats:
    def test_bgr_same_as_gray(self):
        g = _gray(v=128)
        np.testing.assert_array_equal(
            preprocess_crop(g),
            preprocess_crop(np.stack([g, g, g], axis=-1)),
        )

    @pytest.mark.parametrize("h,w", [(8, 8), (64, 64), (200, 300)])
    def test_various_sizes(self, h, w):
        assert preprocess_crop(_gray(h=h, w=w)).shape == (64, 64)


class TestDeterminism:
    def test_same_input_same_output(self):
        img = _checkerboard()
        np.testing.assert_array_equal(preprocess_crop(img), preprocess_crop(img))

    def test_does_not_modify_input(self):
        img = _checkerboard()
        orig = img.copy()
        preprocess_crop(img)
        np.testing.assert_array_equal(img, orig)
