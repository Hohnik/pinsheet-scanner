"""CNN-based pin state classification with test-time augmentation (TTA).

Runs ``TTA_PASSES`` forward passes per crop (first clean, rest with mild
augmentation) and averages sigmoid probabilities before thresholding.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from augment import AugmentConfig, augment
from model import PinClassifier

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
    weights_path: Path, *, device: torch.device | str | None = None,
) -> tuple[PinClassifier, torch.device]:
    """Load a trained PinClassifier from disk."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier not found: {weights_path}")
    dev = resolve_device(device)
    model = PinClassifier()
    try:
        model.load_state_dict(torch.load(weights_path, map_location=dev, weights_only=True))
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot load weights from {weights_path}. "
            "Architecture may have changed — retrain with `just train`."
        ) from e
    model.to(dev).eval()
    return model, dev


def preprocess_crop(crop: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Grayscale → resize → Otsu binarise → [0, 1] float32."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0


@torch.no_grad()
def classify_pins_batch(
    model: PinClassifier,
    crops: list[np.ndarray],
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[tuple[list[int], float]]:
    """Classify pin states for a batch of crops with TTA.

    Returns ``[(pins, confidence), …]`` per crop.
    """
    if not crops:
        return []
    if device is None:
        device = next(model.parameters()).device

    rng = np.random.default_rng(42)
    acc: torch.Tensor | None = None
    for i in range(TTA_PASSES):
        if i == 0:
            arrays = [preprocess_crop(c) for c in crops]
        else:
            grays = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c for c in crops]
            arrays = [preprocess_crop(augment(g, rng, _TTA_CFG)) for g in grays]
        probs = torch.sigmoid(model(torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)))
        acc = probs if acc is None else acc + probs

    avg = acc / TTA_PASSES  # type: ignore[operator]
    conf = (avg - 0.5).abs().mean(dim=1) * 2.0
    return [
        ((avg[i] >= threshold).int().cpu().tolist(), float(conf[i]))
        for i in range(avg.size(0))
    ]
