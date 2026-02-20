"""Pin classifier: shared-backbone CNN with spatial pin extraction + global context.

Input: B×1×64×64 grayscale, pixel values in [0, 1].
Output: B×9 raw logits (apply sigmoid for per-pin probabilities).

Pin layout (diamond pattern, front pin = 0)::

        8
      6   7
    3   4   5
      1   2
        0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PINS: int = 9
PATCH_SIZE: int = 16  # 16×16 patch from feature map → ~22×22 pixel receptive field

PIN_COORDS: list[tuple[int, int]] = [
    (32, 56),                          # 0 – front
    (20, 44), (44, 44),                # 1, 2
    (8,  32), (32, 32), (56, 32),      # 3, 4, 5
    (20, 20), (44, 20),                # 6, 7
    (32,  8),                          # 8 – back
]
"""(cx, cy) of each pin centre in the 64×64 canonical crop."""


class PinClassifier(nn.Module):
    """Shared-backbone CNN combining per-pin local features with global context.

    Three conv layers build a 64-channel feature map at full resolution.
    A 16×16 patch is extracted around each of the 9 known pin positions
    and average-pooled to a 64-dim local vector.  A global average pool
    of the whole feature map gives a 64-dim context vector.  The two are
    concatenated and fed to a 2-layer shared head.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        # local(64) + global(64) → 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                          # B×64×64×64
        h = PATCH_SIZE // 2
        # Per-pin local features
        local = torch.stack([
            F.adaptive_avg_pool2d(feat[:, :, cy - h:cy + h, cx - h:cx + h], 1).flatten(1)
            for cx, cy in PIN_COORDS
        ], dim=1)                                        # B×9×64
        # Global context (broadcast to all pins)
        ctx = (F.adaptive_avg_pool2d(feat, 1)
               .flatten(1).unsqueeze(1).expand(-1, NUM_PINS, -1))  # B×9×64
        return self.head(torch.cat([local, ctx], dim=-1)).squeeze(-1)  # B×9
