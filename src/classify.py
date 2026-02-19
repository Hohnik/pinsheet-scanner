"""CNN-based pin state classification from cropped pin diagrams.

Each diagram shows 9 pins in a diamond layout::

        8
      6   7
    3   4   5
      1   2
        0

Index 0 is the front (nearest) pin; numbering increases toward the back,
left-to-right within each row.  A filled/dark dot means the pin was knocked
down; an empty/light ring means it is still standing.

Inference uses **test-time augmentation** (TTA): each crop is run through
the model ``TTA_PASSES`` times with mild random augmentations, and the
sigmoid probabilities are averaged before thresholding.  The first pass is
always clean (no augmentation) to preserve the most reliable prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from augment import AugmentConfig, augment
from model import PinClassifier, SpatialPinClassifier

# Number of TTA forward passes (first is always clean).
TTA_PASSES: int = 5

# Mild augmentation used for TTA passes 2–N.
_TTA_CFG = AugmentConfig(
    brightness_range=(-20, 20),
    noise_sigma_range=(1.0, 5.0),
    blur_kernels=(0, 0, 3),
    blur_sigma_range=(0.3, 0.8),
    max_rotation_deg=3.0,
    scale_range=(0.95, 1.05),
    grid_line_probability=0.0,
)

# Union type accepted by all public functions in this module.
AnyClassifier = PinClassifier | SpatialPinClassifier


def resolve_device(device: torch.device | str | None) -> torch.device:
    """Pick the best available device when *device* is ``None``."""
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classifier(
    weights_path: Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[AnyClassifier, torch.device]:
    """Load a trained classifier from disk.

    Reads the model bundle sidecar (``<weights>.json``) if present to
    determine the architecture.  Falls back to :class:`~model.PinClassifier`
    for weights saved without a sidecar (backward compatible).

    Args:
        weights_path: Path to a ``.pt`` state-dict file.
        device: Device to load onto.  ``None`` → auto-detect.

    Returns:
        ``(model, device)`` ready for inference.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found at {weights_path}. "
            "Train a model first (see `pinsheet-scanner train`) or pass --classifier."
        )

    resolved = resolve_device(device)

    bundle_path = weights_path.with_suffix(".json")
    arch = "PinClassifier"
    if bundle_path.exists():
        arch = json.loads(bundle_path.read_text()).get("architecture", arch)

    model: AnyClassifier
    if arch == "SpatialPinClassifier":
        model = SpatialPinClassifier()
    else:
        model = PinClassifier()

    state = torch.load(weights_path, map_location=resolved, weights_only=True)
    model.load_state_dict(state)
    model.to(resolved)
    model.eval()
    return model, resolved


def preprocess_crop(
    crop: np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Convert a raw crop to a normalised float32 array ready for the CNN.

    Steps: grayscale → resize → Otsu binarise → [0, 1] float32.

    Args:
        crop: Grayscale or BGR image.
        size: Target ``(width, height)``.  Defaults to ``(64, 64)``.

    Returns:
        Float32 array in [0, 1] with shape ``(height, width)``.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    w, h = size
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0


def _preprocess_with_tta(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment a raw crop then preprocess it for one TTA pass."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return preprocess_crop(augment(gray, rng, _TTA_CFG))


def _confidence_from_probs(probs: torch.Tensor) -> float:
    """Mean distance from the 0.5 decision boundary, scaled to [0, 1]."""
    return float(max(0.0, min(1.0, (probs - 0.5).abs().mean().item() * 2.0)))


@torch.no_grad()
def classify_pins_batch_with_confidence(
    model: AnyClassifier,
    crops: list[np.ndarray],
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[tuple[list[int], float]]:
    """Classify pin states for a batch of cropped diagrams with confidence.

    Uses test-time augmentation (TTA): runs ``TTA_PASSES`` forward passes
    with mild augmentation and averages the sigmoid probabilities.

    Args:
        model: Loaded classifier (``PinClassifier`` or ``SpatialPinClassifier``).
        crops: List of raw grayscale or BGR crops.
        device: Device the model lives on.  ``None`` → inferred from model.
        threshold: Sigmoid probability threshold for binary decision.

    Returns:
        List of ``(pins, confidence)`` tuples, one per crop.
    """
    if not crops:
        return []

    if device is None:
        device = next(model.parameters()).device

    rng = np.random.default_rng(42)
    acc: torch.Tensor | None = None
    for pass_idx in range(TTA_PASSES):
        if pass_idx == 0:
            arrays = [preprocess_crop(c) for c in crops]
        else:
            arrays = [_preprocess_with_tta(c, rng) for c in crops]

        batch = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)
        probs = torch.sigmoid(model(batch))  # (B, 9)
        acc = probs if acc is None else acc + probs

    avg = acc / TTA_PASSES  # type: ignore[operator]

    return [
        (
            (avg[i] >= threshold).int().cpu().tolist(),
            _confidence_from_probs(avg[i]),
        )
        for i in range(avg.size(0))
    ]
