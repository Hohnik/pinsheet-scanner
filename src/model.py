"""CNN models for classifying 9-pin states from cropped pin diagrams.

* ``PinClassifier`` — legacy global-average-pool model.
* ``SpatialPinClassifier`` — spatially-grounded model that extracts a
  fixed patch at each known pin position.  Preferred for new training.

Input: 1 × 64 × 64 grayscale, pixel values in [0, 1].
Output: 9 raw logits (apply sigmoid for per-pin probabilities).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import NUM_PINS, PATCH_SIZE, PIN_COORDS_64


class PinClassifier(nn.Module):
    """Legacy global-pooling CNN — kept for backward compatibility."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for in_ch, out_ch in [(1, 32), (32, 64), (64, 128), (128, 128)]:
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, NUM_PINS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))


class SpatialPinClassifier(nn.Module):
    """Extracts a patch at each of the 9 pin positions and classifies each."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        half = PATCH_SIZE // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * half * half, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, 1))

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, h = x.size(0), PATCH_SIZE // 2
        patches = []
        for cx, cy in PIN_COORDS_64:
            p = x[:, :, cy - h : cy + h, cx - h : cx + h]
            ph, pw = PATCH_SIZE - p.shape[2], PATCH_SIZE - p.shape[3]
            if ph > 0 or pw > 0:
                p = F.pad(p, (0, pw, 0, ph))
            patches.append(p)
        return torch.stack(patches, dim=1).view(B * NUM_PINS, 1, PATCH_SIZE, PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.head(self.encoder(self._extract_patches(x))).view(B, NUM_PINS)
