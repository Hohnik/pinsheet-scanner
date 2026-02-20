"""Image augmentation for pin diagram training data.

Applies realistic scan-artifact effects during training:
  - Photometric: brightness, gamma, noise, shadow gradient
  - Geometric: rotation, scale, perspective warp, aspect ratio jitter
  - Degradation: Gaussian blur, motion blur, JPEG compression
  - Structural: grid-line remnants
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Parameters for the augmentation pipeline.  Ranges sampled uniformly."""

    # Photometric
    brightness_range: tuple[int, int] = (-25, 25)
    gamma_range: tuple[float, float] = (0.6, 1.8)
    noise_sigma_range: tuple[float, float] = (2.0, 8.0)
    shadow_probability: float = 0.3
    shadow_intensity: tuple[float, float] = (0.7, 1.0)

    # Geometric
    max_rotation_deg: float = 5.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    aspect_jitter: tuple[float, float] = (0.85, 1.15)
    perspective_strength: float = 0.06

    # Degradation
    blur_kernels: tuple[int, ...] = (0, 0, 3, 3, 5)
    blur_sigma_range: tuple[float, float] = (0.4, 1.5)
    motion_blur_probability: float = 0.15
    motion_blur_kernels: tuple[int, ...] = (3, 5, 7)
    jpeg_probability: float = 0.2
    jpeg_quality_range: tuple[int, int] = (40, 85)
    # Structural
    grid_line_probability: float = 0.6
    grid_intensity_range: tuple[int, int] = (100, 170)


DEFAULT_CONFIG = AugmentConfig()


# ── Photometric ────────────────────────────────────────────────────────────


def _apply_brightness(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    lo, hi = cfg.brightness_range
    if lo == 0 and hi == 0:
        return img
    return np.clip(img.astype(np.int16) + int(rng.integers(lo, hi + 1)), 0, 255).astype(np.uint8)


def _apply_gamma(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Power-law contrast shift — simulates different scanner exposure."""
    lo, hi = cfg.gamma_range
    if lo == 1.0 and hi == 1.0:
        return img
    gamma = rng.uniform(lo, hi)
    table = (np.arange(256) / 255.0) ** gamma * 255.0
    return table.astype(np.uint8)[img]


def _apply_noise(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    lo, hi = cfg.noise_sigma_range
    if hi <= 0:
        return img
    sigma = rng.uniform(lo, hi)
    return np.clip(
        img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255
    ).astype(np.uint8)


def _apply_shadow(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Add a linear brightness gradient — simulates uneven lighting."""
    if rng.random() >= cfg.shadow_probability:
        return img
    h, w = img.shape[:2]
    lo, hi = cfg.shadow_intensity
    # Random direction: 0=left-right, 1=top-bottom, 2=diagonal
    direction = int(rng.integers(0, 3))
    start, end = rng.uniform(lo, 1.0), rng.uniform(lo, hi)
    if direction == 0:
        grad = np.linspace(start, end, w, dtype=np.float32)[np.newaxis, :]
    elif direction == 1:
        grad = np.linspace(start, end, h, dtype=np.float32)[:, np.newaxis]
    else:
        gx = np.linspace(start, end, w, dtype=np.float32)
        gy = np.linspace(1.0, rng.uniform(lo, 1.0), h, dtype=np.float32)
        grad = gx[np.newaxis, :] * gy[:, np.newaxis]
    return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)


# ── Geometric ──────────────────────────────────────────────────────────────


def _apply_rotation(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    if cfg.max_rotation_deg <= 0:
        return img
    angle = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h), borderValue=int(img[0, 0]))


def _apply_scale_jitter(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Uniform scale jitter (preserves aspect ratio)."""
    lo, hi = cfg.scale_range
    if lo == 1.0 and hi == 1.0:
        return img
    s = rng.uniform(lo, hi)
    h, w = img.shape[:2]
    nw, nh = max(4, int(w * s)), max(4, int(h * s))
    return cv2.resize(
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_AREA,
    )


def _apply_aspect_jitter(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Independent x/y stretch — simulates camera angle and scan distortion."""
    lo, hi = cfg.aspect_jitter
    if lo == 1.0 and hi == 1.0:
        return img
    sx, sy = rng.uniform(lo, hi), rng.uniform(lo, hi)
    h, w = img.shape[:2]
    nw, nh = max(4, int(w * sx)), max(4, int(h * sy))
    return cv2.resize(
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA),
        (w, h), interpolation=cv2.INTER_AREA,
    )


def _apply_perspective(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Random perspective warp — simulates tilted camera / non-flat paper."""
    if cfg.perspective_strength <= 0:
        return img
    h, w = img.shape[:2]
    s = cfg.perspective_strength
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    jitter = rng.uniform(-s, s, (4, 2)) * np.array([[w, h]], dtype=np.float64)
    dst = (src + jitter).astype(np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, mat, (w, h), borderValue=int(img[0, 0]))


# ── Degradation ────────────────────────────────────────────────────────────


def _apply_blur(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    if not cfg.blur_kernels:
        return img
    k = int(rng.choice(cfg.blur_kernels))
    if k < 3:
        return img
    return cv2.GaussianBlur(img, (k, k), sigmaX=rng.uniform(*cfg.blur_sigma_range))


def _apply_motion_blur(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Directional blur — simulates camera shake."""
    if rng.random() >= cfg.motion_blur_probability:
        return img
    k = int(rng.choice(cfg.motion_blur_kernels))
    kernel = np.zeros((k, k), dtype=np.float32)
    angle = rng.uniform(0, 180)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    cx, cy = k // 2, k // 2
    for i in range(k):
        t = (i - cx)
        x = int(round(cx + t * cos_a))
        y = int(round(cy + t * sin_a))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1)
    return cv2.filter2D(img, -1, kernel)


def _apply_jpeg(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """JPEG encode/decode — simulates re-compression artifacts."""
    if rng.random() >= cfg.jpeg_probability:
        return img
    lo, hi = cfg.jpeg_quality_range
    quality = int(rng.integers(lo, hi + 1))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


# ── Structural ─────────────────────────────────────────────────────────────


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


# ── Pipeline ───────────────────────────────────────────────────────────────

_TRANSFORMS = [
    # Structural first (grid lines drawn on clean image).
    _apply_grid_lines,
    # Geometric transforms (before photometric to avoid interpolation on noise).
    _apply_rotation,
    _apply_scale_jitter,
    _apply_aspect_jitter,
    _apply_perspective,
    # Photometric.
    _apply_brightness,
    _apply_gamma,
    _apply_shadow,
    _apply_noise,
    # Degradation last.
    _apply_blur,
    _apply_motion_blur,
    _apply_jpeg,
]


def augment(
    img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Apply the full augmentation pipeline.  Returns a new array."""
    out = img.copy()
    for fn in _TRANSFORMS:
        out = fn(out, rng, cfg)
    return out
