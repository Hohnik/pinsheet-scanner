"""Shared constants for pin diagram detection and classification.

Pin layout (diamond pattern)::

        0
      1   2
    3   4   5
      6   7
        8
"""

NUM_PINS: int = 9

# CNN classifier input size (width, height).
CLASSIFIER_INPUT_SIZE: tuple[int, int] = (64, 64)
