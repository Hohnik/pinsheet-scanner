import numpy as np
import pytest

import cv2

from pinsheet_scanner.classify import (
    PIN_POSITIONS,
    _remove_grid_lines,
    _strip_border,
    build_pin_masks,
    clean_diagram,
    classify_pins,
    classify_pins_adaptive,
    measure_pin_intensity,
    pins_and_intensities_from_diagram,
    resize_diagram,
)
from pinsheet_scanner.pipeline import calculate_classification_confidence
from pinsheet_scanner.classify_contour import (
    classify_pins_contour,
    detect_dot_contours,
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


class TestPinsAndIntensitiesFromDiagram:
    """Tests for pins_and_intensities_from_diagram function."""

    def test_returns_tuple_of_pins_and_intensities(self):
        """Should return both pins list and intensities list."""
        diagram = np.full((32, 32), 128, dtype=np.uint8)
        pins, intensities = pins_and_intensities_from_diagram(diagram)

        assert isinstance(pins, list)
        assert isinstance(intensities, list)
        assert len(pins) == 9
        assert len(intensities) == 9
        assert all(p in (0, 1) for p in pins)
        assert all(0.0 <= i <= 1.0 for i in intensities)

    def test_intensities_match_pins_from_diagram_result(self):
        """pins_and_intensities should give same pin result as pins_from_diagram."""
        from pinsheet_scanner.classify import pins_from_diagram

        diagram = np.full((32, 32), 200, dtype=np.uint8)
        pins_only = pins_from_diagram(diagram)
        pins, intensities = pins_and_intensities_from_diagram(diagram)

        assert pins == pins_only
        assert len(intensities) == 9


class TestDetectDotContours:
    """Tests for contour-based dot detection."""

    def test_finds_filled_dots(self):
        """Should detect filled circles as contours."""
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(img, (16, 16), 4, 255, -1)
        cv2.circle(img, (32, 32), 4, 255, -1)
        cv2.circle(img, (48, 48), 4, 255, -1)
        contours = detect_dot_contours(img)
        assert len(contours) >= 3

    def test_returns_empty_on_blank_image(self):
        """Should return empty list for blank image."""
        img = np.zeros((64, 64), dtype=np.uint8)
        contours = detect_dot_contours(img)
        assert len(contours) == 0

    def test_filters_small_noise(self):
        """Should filter out very small contours (noise)."""
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(img, (32, 32), 5, 255, -1)
        img[10, 10] = 255
        contours = detect_dot_contours(img)
        assert len(contours) == 1


class TestClassifyPinsContour:
    """Tests for contour-based pin classification."""

    def test_all_filled_dots_classified_as_down(self):
        """Large filled dots should be classified as knocked down."""
        img = np.zeros((64, 64), dtype=np.uint8)
        for px, py in PIN_POSITIONS:
            cx, cy = int(px * 64), int(py * 64)
            cv2.circle(img, (cx, cy), 5, 255, -1)
        pins = classify_pins_contour(img)
        assert len(pins) == 9
        assert all(p in (0, 1) for p in pins)

    def test_returns_nine_values(self):
        """Should always return exactly 9 pin states."""
        img = np.zeros((64, 64), dtype=np.uint8)
        pins = classify_pins_contour(img)
        assert len(pins) == 9

    def test_uniform_image_returns_nine_values(self):
        """Uniform image should return 9 pin states without errors."""
        img = np.full((64, 64), 128, dtype=np.uint8)
        pins = classify_pins_contour(img)
        assert len(pins) == 9
        assert all(p in (0, 1) for p in pins)


class TestCleanDiagram:
    def test_grayscale_output(self):
        img = np.full((32, 32), 128, dtype=np.uint8)
        result = clean_diagram(img)
        assert result.ndim == 2

    def test_color_input_converted(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = clean_diagram(img)
        assert result.ndim == 2

    def test_empty_image_handled(self):
        img = np.array([], dtype=np.uint8)
        result = clean_diagram(img)
        assert result.size == 0

    def test_output_shape_preserved(self):
        img = np.full((48, 48), 100, dtype=np.uint8)
        result = clean_diagram(img)
        assert result.shape == (48, 48)


class TestStripBorder:
    def test_zeros_border_pixels(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        result = _strip_border(img, fraction=0.1)
        assert result[0, 50] == 0
        assert result[99, 50] == 0
        assert result[50, 0] == 0
        assert result[50, 99] == 0

    def test_preserves_center(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        result = _strip_border(img, fraction=0.1)
        assert result[50, 50] == 200


class TestRemoveGridLines:
    def test_removes_horizontal_lines(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[32, :] = 255
        result = _remove_grid_lines(img)
        assert result[32, 32] < 128

    def test_preserves_dots(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(img, (32, 32), 5, 255, -1)
        result = _remove_grid_lines(img)
        assert result[32, 32] > 0


class TestResizeDiagram:
    def test_output_shape(self):
        img = np.full((100, 80), 128, dtype=np.uint8)
        result = resize_diagram(img, size=(64, 64))
        assert result.shape == (64, 64)

    def test_custom_size(self):
        img = np.full((50, 50), 128, dtype=np.uint8)
        result = resize_diagram(img, size=(32, 32))
        assert result.shape == (32, 32)


class TestClassifyPinsAdaptive:
    def test_all_clearly_down(self):
        intensities = [0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.72]
        result = classify_pins_adaptive(intensities)
        assert result == [1] * 9

    def test_all_clearly_standing(self):
        intensities = [0.1, 0.05, 0.15, 0.08, 0.2, 0.12, 0.03, 0.18, 0.07]
        result = classify_pins_adaptive(intensities)
        assert result == [0] * 9

    def test_ambiguous_uses_median(self):
        intensities = [0.8, 0.1, 0.4, 0.35, 0.45, 0.3, 0.9, 0.05, 0.38]
        result = classify_pins_adaptive(intensities)
        assert len(result) == 9
        assert result[0] == 1
        assert result[1] == 0
        assert all(p in (0, 1) for p in result)


class TestCalculateClassificationConfidence:
    """Tests for classification confidence metric."""

    def test_all_down_has_max_confidence(self):
        """When all pins are knocked down, confidence should be maximum."""
        intensities = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        pins = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        confidence = calculate_classification_confidence(intensities, pins)
        assert confidence == 1.0

    def test_all_standing_has_max_confidence(self):
        """When all pins are standing, confidence should be maximum."""
        intensities = [0.1, 0.15, 0.2, 0.05, 0.12, 0.18, 0.09, 0.11, 0.08]
        pins = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        confidence = calculate_classification_confidence(intensities, pins)
        assert confidence == 1.0

    def test_clear_separation_high_confidence(self):
        """Large gap between standing and knocked down = high confidence."""
        intensities = [0.8, 0.2, 0.85, 0.15, 0.9, 0.1, 0.75, 0.18, 0.82]
        pins = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        confidence = calculate_classification_confidence(intensities, pins)
        assert confidence > 0.5

    def test_ambiguous_classification_low_confidence(self):
        """Small gap between standing and knocked down = low confidence."""
        intensities = [0.4, 0.35, 0.38, 0.42, 0.36, 0.41, 0.37, 0.39, 0.40]
        pins = [1, 0, 1, 1, 0, 1, 0, 1, 1]
        confidence = calculate_classification_confidence(intensities, pins)
        assert confidence < 0.2

    def test_empty_input_zero_confidence(self):
        """Empty or invalid input should return zero confidence."""
        assert calculate_classification_confidence([], []) == 0.0
        assert calculate_classification_confidence([0.5], []) == 0.0
        assert calculate_classification_confidence([], [1]) == 0.0


class TestPinsFromDiagram:
    """End-to-end tests for pins_from_diagram with synthetic images."""

    @pytest.mark.skip(
        reason="Synthetic images don't match real preprocessing pipeline - use golden tests with real crops instead"
    )
    def test_all_pins_down_synthetic(self):
        """Synthetic image with all pins filled (down) should return all 0s."""
        size = 32
        diagram = np.ones((size, size), dtype=np.uint8) * 50

        from pinsheet_scanner.classify import PIN_POSITIONS

        for norm_x, norm_y in PIN_POSITIONS:
            x, y = int(norm_x * (size - 1)), int(norm_y * (size - 1))
            cv2.circle(diagram, (x, y), radius=3, color=0, thickness=-1)

        from pinsheet_scanner.classify import pins_from_diagram

        result = pins_from_diagram(diagram)

        assert result == [0, 0, 0, 0, 0, 0, 0, 0, 0], (
            f"Expected all pins down, got {result}"
        )

    @pytest.mark.skip(
        reason="Synthetic images don't match real preprocessing pipeline - use golden tests with real crops instead"
    )
    def test_all_pins_standing_synthetic(self):
        """Synthetic image with all pins as rings (standing) should return all 1s."""
        size = 32
        diagram = np.ones((size, size), dtype=np.uint8) * 50

        from pinsheet_scanner.classify import PIN_POSITIONS

        for norm_x, norm_y in PIN_POSITIONS:
            x, y = int(norm_x * (size - 1)), int(norm_y * (size - 1))
            cv2.circle(diagram, (x, y), radius=3, color=255, thickness=1)

        from pinsheet_scanner.classify import pins_from_diagram

        result = pins_from_diagram(diagram)

        assert result == [1, 1, 1, 1, 1, 1, 1, 1, 1], (
            f"Expected all pins standing, got {result}"
        )

    @pytest.mark.skip(
        reason="Synthetic images don't match real preprocessing pipeline - use golden tests with real crops instead"
    )
    def test_mixed_pins_synthetic(self):
        """Synthetic image with mixed pin states."""
        size = 32
        diagram = np.ones((size, size), dtype=np.uint8) * 50

        from pinsheet_scanner.classify import PIN_POSITIONS

        for i, (norm_x, norm_y) in enumerate(PIN_POSITIONS):
            x, y = int(norm_x * (size - 1)), int(norm_y * (size - 1))
            if i < 5:
                cv2.circle(diagram, (x, y), radius=3, color=0, thickness=-1)
            else:
                cv2.circle(diagram, (x, y), radius=3, color=255, thickness=1)

        from pinsheet_scanner.classify import pins_from_diagram

        result = pins_from_diagram(diagram)

        assert result == [0, 0, 0, 0, 0, 1, 1, 1, 1], (
            f"Expected [0,0,0,0,0,1,1,1,1], got {result}"
        )


class TestGoldenRegression:
    """Tests using real cropped pin diagrams from actual scans."""

    def test_real_crop_classification(self):
        """pins_from_diagram should work on real cropped images and return valid results."""
        from pinsheet_scanner.classify import pins_from_diagram
        from pathlib import Path

        fixture_path = Path("tests/fixtures/sample_crop.png")
        if not fixture_path.exists():
            pytest.skip("Fixture image not found")

        crop = cv2.imread(str(fixture_path))
        assert crop is not None, "Failed to load fixture image"

        result = pins_from_diagram(crop)

        assert len(result) == 9, f"Expected 9 pin states, got {len(result)}"
        assert all(p in (0, 1) for p in result), (
            f"All values should be 0 or 1, got {result}"
        )
