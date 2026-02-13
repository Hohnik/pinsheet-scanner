import numpy as np
import pytest

from pinsheet_scanner.classify import (
    PIN_POSITIONS,
    build_pin_masks,
    classify_pins,
    measure_pin_intensity,
)


class TestBuildPinMasks:
    def test_returns_nine_masks(self):
        masks = build_pin_masks()
        assert len(masks) == 9

    def test_mask_shape_matches_default_size(self):
        masks = build_pin_masks()
        for mask in masks:
            assert mask.shape == (64, 64)

    def test_mask_shape_matches_custom_size(self):
        masks = build_pin_masks(size=(48, 48))
        for mask in masks:
            assert mask.shape == (48, 48)

    def test_masks_are_binary(self):
        masks = build_pin_masks()
        for mask in masks:
            unique = set(np.unique(mask))
            assert unique <= {0, 255}

    def test_each_mask_has_nonzero_pixels(self):
        masks = build_pin_masks()
        for i, mask in enumerate(masks):
            assert np.count_nonzero(mask) > 0, f"Mask {i} is entirely zero"

    def test_masks_do_not_overlap(self):
        masks = build_pin_masks(size=(64, 64), radius=3)
        combined = np.zeros((64, 64), dtype=np.int32)
        for mask in masks:
            combined += (mask > 0).astype(np.int32)
        assert combined.max() <= 1, "Some masks overlap"

    def test_custom_radius(self):
        small = build_pin_masks(radius=2)
        large = build_pin_masks(radius=5)
        small_total = sum(np.count_nonzero(m) for m in small)
        large_total = sum(np.count_nonzero(m) for m in large)
        assert large_total > small_total


class TestMeasurePinIntensity:
    def test_white_image_returns_high_intensity(self):
        img = np.full((64, 64), 255, dtype=np.uint8)
        masks = build_pin_masks()
        intensities = measure_pin_intensity(img, masks)
        assert len(intensities) == 9
        for val in intensities:
            assert val == pytest.approx(1.0)

    def test_black_image_returns_zero_intensity(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        masks = build_pin_masks()
        intensities = measure_pin_intensity(img, masks)
        assert len(intensities) == 9
        for val in intensities:
            assert val == pytest.approx(0.0)

    def test_selective_pin_intensity(self):
        """Paint only the region of pin 0 white; it should have high
        intensity while others stay low."""
        size = (64, 64)
        radius = 5
        img = np.zeros(size, dtype=np.uint8)
        masks = build_pin_masks(size=size, radius=radius)

        # Fill pin-0 region with white
        img[masks[0] > 0] = 255

        intensities = measure_pin_intensity(img, masks)
        assert intensities[0] == pytest.approx(1.0)
        for i in range(1, 9):
            assert intensities[i] == pytest.approx(0.0)


class TestClassifyPins:
    def test_all_down(self):
        intensities = [0.9] * 9
        result = classify_pins(intensities)
        assert result == [1] * 9

    def test_all_standing(self):
        intensities = [0.1] * 9
        result = classify_pins(intensities)
        assert result == [0] * 9

    def test_mixed(self):
        intensities = [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.05, 0.95]
        result = classify_pins(intensities)
        assert result == [1, 0, 1, 0, 1, 0, 1, 0, 1]

    def test_custom_threshold(self):
        intensities = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        high_thresh = classify_pins(intensities, threshold=0.5)
        low_thresh = classify_pins(intensities, threshold=0.2)
        assert high_thresh == [0] * 9
        assert low_thresh == [1] * 9

    def test_returns_list_of_ints(self):
        intensities = [0.5] * 9
        result = classify_pins(intensities, threshold=0.5)
        for val in result:
            assert isinstance(val, int)

    def test_boundary_value_at_threshold(self):
        """Values exactly at the threshold should be classified as down."""
        intensities = [0.5]
        result = classify_pins(intensities, threshold=0.5)
        assert result == [1]


class TestPinPositions:
    def test_nine_positions(self):
        assert len(PIN_POSITIONS) == 9

    def test_values_in_unit_range(self):
        for x, y in PIN_POSITIONS:
            assert 0.0 <= x <= 1.0, f"x={x} out of range"
            assert 0.0 <= y <= 1.0, f"y={y} out of range"

    def test_top_pin_is_centered_horizontally(self):
        x, _ = PIN_POSITIONS[0]
        assert x == pytest.approx(0.5, abs=0.05)
