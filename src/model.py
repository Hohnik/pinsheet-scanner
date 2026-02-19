"""CNN models for classifying 9-pin states from cropped pin diagrams.

Two architectures are provided:

* ``PinClassifier`` — legacy global-average-pool model (backward compatible).
* ``SpatialPinClassifier`` — spatially-grounded model that extracts a fixed
  patch at each known pin position and classifies each independently.
  Preferred for new training runs.

Input for both: 1 × 64 × 64 grayscale image with pixel values in [0, 1].
Output for both: 9 raw logits — apply sigmoid for per-pin probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import NUM_PINS, PATCH_SIZE, PIN_COORDS_64


def _conv_block(in_ch: int, out_ch: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]


class PinClassifier(nn.Module):
    """Legacy global-pooling CNN — kept for backward compatibility.

    Args:
        dropout: Drop probability before the final linear layer.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(1, 32),    # 1×64×64  → 32×32×32
            *_conv_block(32, 64),   #           → 64×16×16
            *_conv_block(64, 128),  #           → 128×8×8
            *_conv_block(128, 128), #           → 128×4×4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, NUM_PINS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class SpatialPinClassifier(nn.Module):
    """Spatial ROI pin classifier using fixed per-pin patch queries.

    For each of the 9 known pin positions in the 64×64 crop, a small
    ``PATCH_SIZE × PATCH_SIZE`` patch is extracted and passed through a
    shared convolutional encoder.  A shared head then maps each encoded
    patch to a single logit.

    Advantages over global-pooling:
    * 9× more training signal per labelled crop.
    * Spatially grounded — looks at exactly the right location per pin.
    * Per-pin task is trivially simple: filled dot vs. empty ring.

    Args:
        dropout: Drop probability before the per-pin output head.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        half = PATCH_SIZE // 2  # 6

        # Shared encoder: (1, P, P) → 32-dim feature vector.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 16 × P  × P
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 16 × P/2 × P/2
            nn.Flatten(),
            nn.Linear(16 * half * half, 32),
            nn.ReLU(inplace=True),
        )

        # Shared per-pin head: one scalar logit per pin.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract one patch per pin, return (B * NUM_PINS, 1, P, P)."""
        B = x.size(0)
        h = PATCH_SIZE // 2
        patches: list[torch.Tensor] = []
        for cx, cy in PIN_COORDS_64:
            patch = x[:, :, cy - h : cy + h, cx - h : cx + h]
            ph = PATCH_SIZE - patch.shape[2]
            pw = PATCH_SIZE - patch.shape[3]
            if ph > 0 or pw > 0:
                patch = F.pad(patch, (0, pw, 0, ph))
            patches.append(patch)
        # (B, 9, 1, P, P) → (B*9, 1, P, P)
        return torch.stack(patches, dim=1).view(B * NUM_PINS, 1, PATCH_SIZE, PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, 9)``."""
        B = x.size(0)
        patches = self._extract_patches(x)      # (B*9, 1, P, P)
        features = self.encoder(patches)         # (B*9, 32)
        logits = self.head(features)             # (B*9, 1)
        return logits.view(B, NUM_PINS)          # (B, 9)
