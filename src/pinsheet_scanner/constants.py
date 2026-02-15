"""Shared constants for pin diagram classification."""

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

DEFAULT_SIZE: tuple[int, int] = (64, 64)
DEFAULT_MASK_RADIUS: int = 5
