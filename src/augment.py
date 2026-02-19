"""Image augmentation for pin diagram training data.

Applies scan-artifact effects (noise, blur, rotation, brightness jitter,
grid-line remnants) to crops during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Parameters for the augmentation pipeline.  Ranges sampled uniformly."""

    brightness_range: tuple[int, int] = (-25, 25)
    noise_sigma_range: tuple[float, float] = (2.0, 8.0)
    blur_kernels: tuple[int, ...] = (0, 3, 3, 5)
    blur_sigma_range: tuple[float, float] = (0.4, 1.5)
    max_rotation_deg: float = 4.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    grid_line_probability: float = 0.6
    grid_intensity_range: tuple[int, int] = (100, 170)


DEFAULT_CONFIG = AugmentConfig()


def _apply_grid_lines(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    if rng.random() >= cfg.grid_line_probability:
        return img
    h, w = img.shape[:2]
    intensity = int(rng.integers(*cfg.grid_intensity_range))
    margin_y, margin_x = max(3, h // 8), max(3, w // 8)
    for y in (int(rng.integers(0, margin_y)), int(rng.integers(h - margin_y, h))):
        if rng.random() < 0.5:
            cv2.line(img, (0, y), (w, y), intensity, 1)
    for x in (int(rng.integers(0, margin_x)), int(rng.integers(w - margin_x, w))):
        if rng.random() < 0.5:
            cv2.line(img, (x, 0), (x, h), intensity, 1)
    return img


def _apply_brightness(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    lo, hi = cfg.brightness_range
    if lo == 0 and hi == 0:
        return img
    return np.clip(img.astype(np.int16) + int(rng.integers(lo, hi + 1)), 0, 255).astype(np.uint8)


def _apply_noise(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    lo, hi = cfg.noise_sigma_range
    if hi <= 0:
        return img
    return np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(lo, hi), img.shape), 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    if not cfg.blur_kernels:
        return img
    k = int(rng.choice(cfg.blur_kernels))
    if k < 3:
        return img
    return cv2.GaussianBlur(img, (k, k), sigmaX=rng.uniform(*cfg.blur_sigma_range))


def _apply_rotation(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    if cfg.max_rotation_deg <= 0:
        return img
    angle = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h), borderValue=int(img[0, 0]))


def _apply_scale_jitter(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    lo, hi = cfg.scale_range
    if lo == 1.0 and hi == 1.0:
        return img
    h, w = img.shape[:2]
    nw, nh = max(4, int(w * rng.uniform(lo, hi))), max(4, int(h * rng.uniform(lo, hi)))
    return cv2.resize(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_AREA)


_TRANSFORMS = [
    _apply_grid_lines, _apply_brightness, _apply_noise,
    _apply_blur, _apply_rotation, _apply_scale_jitter,
]


def augment(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Apply the full augmentation pipeline.  Returns a new array."""
    out = img.copy()
    for fn in _TRANSFORMS:
        out = fn(out, rng, cfg)
    return out
