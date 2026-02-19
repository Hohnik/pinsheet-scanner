"""Shared constants for pin diagram detection and classification.

Pin layout (diamond pattern)::

        8
      6   7
    3   4   5
      1   2
        0

Index 0 is the front (nearest) pin; numbering increases toward the back,
left-to-right within each row.
"""

NUM_PINS: int = 9

# CNN classifier input size (width, height).
CLASSIFIER_INPUT_SIZE: tuple[int, int] = (64, 64)

# Patch size used by SpatialPinClassifier (pixels, square).
PATCH_SIZE: int = 12

# (cx, cy) of each pin centre in the 64×64 canonical crop.
# Origin is top-left; y increases downward. Rows are spaced 12 px apart.
# Numbering: 0 = front/bottom, 8 = back/top.
PIN_COORDS_64: list[tuple[int, int]] = [
    (32, 56),            # 0 – front / bottom
    (20, 44), (44, 44),  # 1, 2
    (8,  32), (32, 32), (56, 32),  # 3, 4, 5
    (20, 20), (44, 20),  # 6, 7
    (32,  8),            # 8 – back / top
]
