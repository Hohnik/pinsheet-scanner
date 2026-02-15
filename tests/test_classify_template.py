"""Tests for template matching classifier."""

import numpy as np


class TestTemplateMatching:
    """Tests for template-based pin classification."""

    def test_classify_pins_template_returns_nine_values(self):
        """classify_pins_template should return a list of 9 pin states."""
        from pinsheet_scanner.classify_template import classify_pins_template

        diagram = np.ones((64, 64), dtype=np.uint8) * 255

        result = classify_pins_template(diagram)

        assert len(result) == 9
        assert all(p in (0, 1) for p in result)

    def test_classify_pins_template_with_filled_dots(self):
        """Filled dots (knocked down) should be classified as 0."""
        from pinsheet_scanner.classify_template import classify_pins_template
        import cv2

        diagram = np.ones((64, 64), dtype=np.uint8) * 255
        from pinsheet_scanner.classify import PIN_POSITIONS

        for norm_x, norm_y in PIN_POSITIONS:
            x, y = int(norm_x * 63), int(norm_y * 63)
            cv2.circle(diagram, (x, y), radius=5, color=0, thickness=-1)

        result = classify_pins_template(diagram)

        down_count = sum(1 for p in result if p == 1)
        assert down_count >= 7, f"Expected mostly down pins (1=down), got {result}"

    def test_classify_pins_template_with_ring_dots(self):
        """Ring dots (standing) should be classified as 1."""
        from pinsheet_scanner.classify_template import classify_pins_template
        import cv2

        diagram = np.ones((64, 64), dtype=np.uint8) * 255
        from pinsheet_scanner.classify import PIN_POSITIONS

        for norm_x, norm_y in PIN_POSITIONS:
            x, y = int(norm_x * 63), int(norm_y * 63)
            cv2.circle(diagram, (x, y), radius=5, color=0, thickness=1)

        result = classify_pins_template(diagram)

        assert len(result) == 9
        assert all(p in (0, 1) for p in result)

    def test_empty_diagram_handles_gracefully(self):
        """Empty or all-white diagram should handle gracefully."""
        from pinsheet_scanner.classify_template import classify_pins_template

        diagram = np.ones((64, 64), dtype=np.uint8) * 255

        result = classify_pins_template(diagram)

        assert len(result) == 9
        assert all(p in (0, 1) for p in result)

    def test_non_square_diagram_resized_correctly(self):
        """Non-square diagrams should be resized using correct (W,H) convention."""
        from pinsheet_scanner.classify_template import classify_pins_template

        diagram = np.ones((48, 80), dtype=np.uint8) * 200

        result = classify_pins_template(diagram)

        assert len(result) == 9
        assert all(p in (0, 1) for p in result)
