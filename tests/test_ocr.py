"""Tests for the OCR module."""

import cv2
import numpy as np

from pinsheet_scanner.detect import Detection
from pinsheet_scanner.ocr import extract_row_scores


class TestExtractRowScores:
    """Tests for extract_row_scores function."""

    def test_extract_single_digit_from_synthetic_image(self):
        """extract_row_scores should recognize a printed digit."""
        image = np.zeros((200, 400), dtype=np.uint8)
        cv2.putText(
            image,
            "7",
            (155, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            255,
            3,
            cv2.LINE_AA,
        )

        detections = [
            Detection(x_center=250, y_center=100, width=100, height=100, confidence=0.9)
        ]

        scores = extract_row_scores(image, detections)

        assert len(scores) == 1
        assert scores[0] == 7

    def test_returns_none_for_unrecognizable_text(self):
        """extract_row_scores should return None when no digit is found."""
        image = np.zeros((200, 400), dtype=np.uint8)
        cv2.putText(
            image, "X", (155, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3, cv2.LINE_AA
        )

        detections = [
            Detection(x_center=250, y_center=100, width=100, height=100, confidence=0.9)
        ]

        scores = extract_row_scores(image, detections)

        assert len(scores) == 1
        assert scores[0] is None

    def test_handles_multiple_detections(self):
        """extract_row_scores should process multiple rows."""
        image = np.zeros((400, 400), dtype=np.uint8)
        cv2.putText(
            image, "5", (155, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3, cv2.LINE_AA
        )
        cv2.putText(
            image, "9", (155, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3, cv2.LINE_AA
        )

        detections = [
            Detection(
                x_center=250, y_center=100, width=100, height=100, confidence=0.9
            ),
            Detection(
                x_center=250, y_center=300, width=100, height=100, confidence=0.9
            ),
        ]

        scores = extract_row_scores(image, detections)

        assert len(scores) == 2
        assert scores[0] == 5
        assert scores[1] == 9
