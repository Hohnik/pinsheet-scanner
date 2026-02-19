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

PATCH_SIZE: int = 12
"""Patch size used by SpatialPinClassifier (pixels, square)."""

PIN_COORDS_64: list[tuple[int, int]] = [
    (32, 56),                          # 0 – front / bottom
    (20, 44), (44, 44),                # 1, 2
    (8,  32), (32, 32), (56, 32),      # 3, 4, 5
    (20, 20), (44, 20),                # 6, 7
    (32,  8),                          # 8 – back / top
]
"""(cx, cy) of each pin centre in the 64×64 canonical crop."""
