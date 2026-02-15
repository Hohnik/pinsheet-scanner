"""Shared constants for pin diagram detection and classification."""

# Diamond layout: 9 pin positions as (x, y) in normalised [0, 1] coordinates.
#
#       0
#     1   2
#   3   4   5
#     6   7
#       8
#
PIN_POSITIONS: list[tuple[float, float]] = [
    (0.50, 0.10),
    (0.30, 0.30),
    (0.70, 0.30),
    (0.10, 0.50),
    (0.50, 0.50),
    (0.90, 0.50),
    (0.30, 0.70),
    (0.70, 0.70),
    (0.50, 0.90),
]

NUM_PINS: int = 9

# CNN classifier input size (width, height).
CLASSIFIER_INPUT_SIZE: tuple[int, int] = (64, 64)
