"""CNN-based pin state classification from cropped pin diagrams.

Each diagram shows 9 pins in a diamond layout:

        0
      1   2
    3   4   5
      6   7
        8

A filled/dark dot means the pin was knocked down; an empty/light ring means
it is still standing.

This module loads a trained ``PinClassifier`` CNN and runs inference on
raw grayscale crops produced by the YOLO detection stage.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .constants import CLASSIFIER_INPUT_SIZE
from .model import PinClassifier

__all__ = [
    "load_classifier",
    "classify_pins",
    "classify_pins_batch",
    "classify_pins_with_confidence",
    "classify_pins_batch_with_confidence",
]


def _resolve_device(device: torch.device | str | None) -> torch.device:
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
) -> tuple[PinClassifier, torch.device]:
    """Load a trained PinClassifier from disk.

    Args:
        weights_path: Path to a ``.pt`` state-dict file.
        device: Device to load onto.  ``None`` → auto-detect.

    Returns:
        Tuple of ``(model, device)`` ready for inference.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found at {weights_path}. "
            "Train a model first (see scripts/train_classifier.py) or pass --classifier-model."
        )

    resolved = _resolve_device(device)
    model = PinClassifier()
    state = torch.load(weights_path, map_location=resolved, weights_only=True)
    model.load_state_dict(state)
    model.to(resolved)
    model.eval()
    return model, resolved


def _preprocess(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Convert a raw crop to a normalised float32 array ready for the CNN.

    Steps: grayscale → resize → Otsu binarise → [0, 1] float32.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    w, h = size
    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(np.float32) / 255.0


def _confidence_from_probs(probs: torch.Tensor) -> float:
    """Mean distance from the 0.5 decision boundary, scaled to [0, 1]."""
    return float(max(0.0, min(1.0, (probs - 0.5).abs().mean().item() * 2.0)))


# ---------------------------------------------------------------------------
# Core: batch inference with confidence (all other variants delegate here)
# ---------------------------------------------------------------------------


@torch.no_grad()
def classify_pins_batch_with_confidence(
    model: PinClassifier,
    crops: list[np.ndarray],
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[tuple[list[int], float]]:
    """Classify pin states for a batch of cropped diagrams with confidence.

    Args:
        model: Loaded ``PinClassifier``.
        crops: List of raw grayscale or BGR crops.
        device: Device the model lives on.
        threshold: Sigmoid probability threshold.

    Returns:
        List of ``(pins, confidence)`` tuples, one per crop.
    """
    if not crops:
        return []

    if device is None:
        device = next(model.parameters()).device

    arrays = [_preprocess(c, CLASSIFIER_INPUT_SIZE) for c in crops]
    batch = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)
    probs = torch.sigmoid(model(batch))  # (B, 9)

    return [
        (
            (probs[i] >= threshold).int().cpu().tolist(),
            _confidence_from_probs(probs[i]),
        )
        for i in range(probs.size(0))
    ]


# ---------------------------------------------------------------------------
# Thin wrappers
# ---------------------------------------------------------------------------


def classify_pins_batch(
    model: PinClassifier,
    crops: list[np.ndarray],
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[list[int]]:
    """Classify pin states for a batch of crops (without confidence)."""
    return [
        pins
        for pins, _ in classify_pins_batch_with_confidence(
            model,
            crops,
            device=device,
            threshold=threshold,
        )
    ]


def classify_pins_with_confidence(
    model: PinClassifier,
    crop: np.ndarray,
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> tuple[list[int], float]:
    """Classify the 9 pin states in a single crop, returning confidence."""
    return classify_pins_batch_with_confidence(
        model,
        [crop],
        device=device,
        threshold=threshold,
    )[0]


def classify_pins(
    model: PinClassifier,
    crop: np.ndarray,
    *,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> list[int]:
    """Classify the 9 pin states in a single cropped diagram."""
    pins, _ = classify_pins_with_confidence(
        model,
        crop,
        device=device,
        threshold=threshold,
    )
    return pins
