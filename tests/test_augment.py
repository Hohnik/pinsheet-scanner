"""Tests for the augmentation pipeline."""

import numpy as np

from augment import AugmentConfig, augment


def _img(h=40, w=50, v=200):
    return np.full((h, w), v, dtype=np.uint8)


class TestAugment:
    def test_preserves_shape_and_dtype(self):
        img = _img()
        out = augment(img, np.random.default_rng(0))
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_does_not_modify_input(self):
        img = _img()
        original = img.copy()
        augment(img, np.random.default_rng(0))
        np.testing.assert_array_equal(img, original)

    def test_output_in_valid_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            out = augment(_img(v=128), rng)
            assert 0 <= out.min() and out.max() <= 255

    def test_noop_config_preserves_pixels(self):
        cfg = AugmentConfig(brightness_range=(0, 0), noise_sigma_range=(0, 0),
                            blur_kernels=(), max_rotation_deg=0, scale_range=(1, 1),
                            grid_line_probability=0, gamma_range=(1.0, 1.0),
                            cutout_probability=0.0)
        np.testing.assert_array_equal(_img(), augment(_img(), np.random.default_rng(0), cfg))

    def test_reproducible_with_same_seed(self):
        img = _img(v=128)
        a = augment(img, np.random.default_rng(99))
        b = augment(img, np.random.default_rng(99))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        img = _img(v=128)
        assert not np.array_equal(
            augment(img, np.random.default_rng(0)),
            augment(img, np.random.default_rng(1)),
        )
