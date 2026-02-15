"""Composable image augmentation for pin diagram training data.

Applies realistic scan-artifact effects (noise, blur, rotation, brightness
jitter, grid-line remnants) to real crop images during training so that
the classifier generalises across scan quality and printing variation.
Each transform is a pure function ``(image, rng) → image`` which makes
the pipeline easy to test and extend.

Usage::

    from pinsheet_scanner.augment import augment, AugmentConfig

    rng = np.random.default_rng(42)
    noisy = augment(clean_image, rng)                       # defaults
    noisy = augment(clean_image, rng, AugmentConfig(max_rotation_deg=8))
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    """Parameters controlling the augmentation pipeline.

    Every range is specified as ``(lo, hi)`` and sampled uniformly per
    image.  Setting a range to ``(0, 0)`` disables that transform.

    Attributes:
        brightness_range: Additive brightness shift (integer pixels).
        noise_sigma_range: Std-dev of additive Gaussian noise.
        blur_kernels: Candidate Gaussian-blur kernel sizes (0 = skip).
        blur_sigma_range: Sigma range for Gaussian blur.
        max_rotation_deg: Maximum rotation in either direction (degrees).
        scale_range: Multiplicative scale jitter per axis.
        grid_line_probability: Chance of drawing faint grid-line remnants.
        grid_intensity_range: Gray value range for grid lines.
    """

    brightness_range: tuple[int, int] = (-25, 25)
    noise_sigma_range: tuple[float, float] = (2.0, 8.0)
    blur_kernels: tuple[int, ...] = (0, 3, 3, 5)
    blur_sigma_range: tuple[float, float] = (0.4, 1.5)
    max_rotation_deg: float = 4.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    grid_line_probability: float = 0.6
    grid_intensity_range: tuple[int, int] = (100, 170)


DEFAULT_CONFIG = AugmentConfig()


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def _apply_grid_lines(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Draw faint horizontal/vertical lines near the edges (grid remnants)."""
    if rng.random() >= cfg.grid_line_probability:
        return img

    h, w = img.shape[:2]
    lo, hi = cfg.grid_intensity_range
    intensity = int(rng.integers(lo, hi + 1))
    margin_y = max(3, h // 8)
    margin_x = max(3, w // 8)

    # Up to two horizontal and two vertical lines near the edges.
    for y in (int(rng.integers(0, margin_y)), int(rng.integers(h - margin_y, h))):
        if rng.random() < 0.5:
            cv2.line(img, (0, y), (w, y), intensity, 1)
    for x in (int(rng.integers(0, margin_x)), int(rng.integers(w - margin_x, w))):
        if rng.random() < 0.5:
            cv2.line(img, (x, 0), (x, h), intensity, 1)

    return img


def _apply_brightness(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Shift all pixel values by a random offset."""
    lo, hi = cfg.brightness_range
    if lo == 0 and hi == 0:
        return img
    shift = int(rng.integers(lo, hi + 1))
    return np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)


def _apply_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Add pixel-wise Gaussian noise."""
    lo, hi = cfg.noise_sigma_range
    if hi <= 0:
        return img
    sigma = rng.uniform(lo, hi)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_blur(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Apply Gaussian blur with a randomly chosen kernel size."""
    if not cfg.blur_kernels:
        return img
    ksize = int(rng.choice(cfg.blur_kernels))
    if ksize < 3:
        return img
    lo, hi = cfg.blur_sigma_range
    sigma = rng.uniform(lo, hi)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def _apply_rotation(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Rotate the image by a small random angle."""
    if cfg.max_rotation_deg <= 0:
        return img
    angle = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = int(img[0, 0])  # fill with top-left pixel colour
    return cv2.warpAffine(img, mat, (w, h), borderValue=border)


def _apply_scale_jitter(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> np.ndarray:
    """Resize to a slightly different aspect ratio and back."""
    lo, hi = cfg.scale_range
    if lo == 1.0 and hi == 1.0:
        return img
    h, w = img.shape[:2]
    new_w = max(4, int(w * rng.uniform(lo, hi)))
    new_h = max(4, int(h * rng.uniform(lo, hi)))
    tmp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(tmp, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

# Ordered list of transforms applied during augmentation.
_TRANSFORMS = [
    _apply_grid_lines,
    _apply_brightness,
    _apply_noise,
    _apply_blur,
    _apply_rotation,
    _apply_scale_jitter,
]


def augment(
    img: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Apply the full augmentation pipeline to a grayscale image.

    The input is **not** modified in-place; a new array is returned.

    Args:
        img: Grayscale ``uint8`` image (H×W).
        rng: NumPy random generator for reproducibility.
        cfg: Augmentation parameters.

    Returns:
        Augmented grayscale ``uint8`` image with the same shape as *img*.
    """
    out = img.copy()
    for transform in _TRANSFORMS:
        out = transform(out, rng, cfg)
    return out
