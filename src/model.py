"""Tiny CNN for classifying 9-pin states from cropped pin diagrams.

Input:  1×64×64 grayscale image (pixel values in [0, 1])
Output: 9 logits — apply sigmoid for independent per-pin probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from constants import NUM_PINS


def _conv_block(in_ch: int, out_ch: int) -> list[nn.Module]:
    """Return layers for one conv block (Conv → BN → ReLU → Pool)."""
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]


class PinClassifier(nn.Module):
    """Tiny CNN that predicts 9 independent pin states from a diagram crop.

    Args:
        dropout: Drop probability applied before the final linear layer.
            Defaults to 0.3.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 32),  # 1×64×64 → 32×32×32
            *_conv_block(32, 64),  # → 64×16×16
            *_conv_block(64, 128),  # → 128×8×8
            *_conv_block(128, 128),  # → 128×4×4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, NUM_PINS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 9)``."""
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
