"""CNN-based pin state classification with test-time augmentation (TTA).

Inference runs ``TTA_PASSES`` forward passes per crop (first clean, rest
with mild augmentation) and averages sigmoid probabilities before thresholding.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from augment import AugmentConfig, augment
from model import PinClassifier, SpatialPinClassifier

TTA_PASSES: int = 5

_TTA_CFG = AugmentConfig(
    brightness_range=(-20, 20),
    noise_sigma_range=(1.0, 5.0),
    blur_kernels=(0, 0, 3),
    blur_sigma_range=(0.3, 0.8),
    max_rotation_deg=3.0,
    scale_range=(0.95, 1.05),
    grid_line_probability=0.0,
)


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
) -> tuple[PinClassifier | SpatialPinClassifier, torch.device]:
    """Load a trained classifier from disk.

    Reads the JSON sidecar (``<weights>.json``) to pick the architecture.
    Falls back to ``PinClassifier`` for weights without a sidecar.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found at {weights_path}. "
            "Train a model first (see `pinsheet-scanner train`)."
        )
    resolved = resolve_device(device)

    bundle_path = weights_path.with_suffix(".json")
    arch = "PinClassifier"
    if bundle_path.exists():
        arch = json.loads(bundle_path.read_text()).get("architecture", arch)

    model: PinClassifier | SpatialPinClassifier
    model = SpatialPinClassifier() if arch == "SpatialPinClassifier" else PinClassifier()

    model.load_state_dict(torch.load(weights_path, map_location=resolved, weights_only=True))
    model.to(resolved).eval()
    return model, resolved


def preprocess_crop(crop: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Grayscale → resize → Otsu binarise → [0, 1] float32."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0


def _preprocess_with_tta(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return preprocess_crop(augment(gray, rng, _TTA_CFG))


@torch.no_grad()
def classify_pins_batch(
    model: PinClassifier | SpatialPinClassifier,
    crops: list[np.ndarray],
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[tuple[list[int], float]]:
    """Classify pin states for a batch of crops with TTA.

    Returns list of ``(pins, confidence)`` tuples, one per crop.
    """
    if not crops:
        return []
    if device is None:
        device = next(model.parameters()).device

    rng = np.random.default_rng(42)
    acc: torch.Tensor | None = None
    for i in range(TTA_PASSES):
        arrays = [preprocess_crop(c) for c in crops] if i == 0 else [_preprocess_with_tta(c, rng) for c in crops]
        probs = torch.sigmoid(model(torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)))
        acc = probs if acc is None else acc + probs

    avg = acc / TTA_PASSES  # type: ignore[operator]
    return [
        (
            (avg[i] >= threshold).int().cpu().tolist(),
            float((avg[i] - 0.5).abs().mean().item() * 2.0),
        )
        for i in range(avg.size(0))
    ]
