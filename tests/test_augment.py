"""Tests for the image augmentation pipeline."""

import numpy as np
import pytest

from augment import AugmentConfig, augment


def _test_image(h: int = 40, w: int = 50, value: int = 200) -> np.ndarray:
    """Create a simple grayscale test image."""
    return np.full((h, w), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# augment — basic contracts
# ---------------------------------------------------------------------------


class TestAugment:
    def test_preserves_shape(self):
        img = _test_image()
        out = augment(img, np.random.default_rng(0))
        assert out.shape == img.shape

    def test_returns_uint8(self):
        img = _test_image()
        out = augment(img, np.random.default_rng(0))
        assert out.dtype == np.uint8

    def test_does_not_modify_input(self):
        img = _test_image()
        original = img.copy()
        augment(img, np.random.default_rng(0))
        np.testing.assert_array_equal(img, original)

    def test_output_within_valid_range(self):
        img = _test_image(value=128)
        rng = np.random.default_rng(42)
        for _ in range(20):
            out = augment(img, rng)
            assert out.min() >= 0
            assert out.max() <= 255

    def test_works_on_small_image(self):
        img = _test_image(h=8, w=8, value=100)
        out = augment(img, np.random.default_rng(0))
        assert out.shape == (8, 8)

    def test_works_on_large_image(self):
        img = _test_image(h=256, w=256, value=180)
        out = augment(img, np.random.default_rng(0))
        assert out.shape == (256, 256)


# ---------------------------------------------------------------------------
# AugmentConfig — frozen dataclass & disabling transforms
# ---------------------------------------------------------------------------


class TestAugmentConfig:
    def test_is_frozen(self):
        cfg = AugmentConfig()
        with pytest.raises(AttributeError):
            cfg.max_rotation_deg = 99  # type: ignore[misc]

    def test_no_augment_yields_copy(self):
        """With everything zeroed out, the result should equal the input."""
        cfg = AugmentConfig(
            brightness_range=(0, 0),
            noise_sigma_range=(0.0, 0.0),
            blur_kernels=(),
            max_rotation_deg=0.0,
            scale_range=(1.0, 1.0),
            grid_line_probability=0.0,
        )
        img = _test_image()
        out = augment(img, np.random.default_rng(0), cfg)
        np.testing.assert_array_equal(img, out)

    def test_default_values_are_sensible(self):
        cfg = AugmentConfig()
        lo, hi = cfg.brightness_range
        assert lo < 0 < hi
        assert cfg.max_rotation_deg > 0
        assert cfg.noise_sigma_range[1] > 0

    def test_custom_config_applies(self):
        """Strong augmentation should visibly change the image."""
        cfg = AugmentConfig(
            brightness_range=(-50, 50),
            noise_sigma_range=(10.0, 20.0),
            blur_kernels=(5, 7),
            max_rotation_deg=15.0,
            scale_range=(0.7, 1.3),
            grid_line_probability=1.0,
        )
        img = _test_image()
        out = augment(img, np.random.default_rng(0), cfg)
        assert not np.array_equal(img, out)


# ---------------------------------------------------------------------------
# augment — reproducibility
# ---------------------------------------------------------------------------


class TestAugmentReproducibility:
    def test_same_seed_same_output(self):
        img = _test_image(value=128)
        a = augment(img, np.random.default_rng(99))
        b = augment(img, np.random.default_rng(99))
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_output(self):
        img = _test_image(value=128)
        a = augment(img, np.random.default_rng(0))
        b = augment(img, np.random.default_rng(1))
        assert not np.array_equal(a, b)
