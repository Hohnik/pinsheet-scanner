"""Training: dataset, training loop, and hyperparameter management."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from augment import AugmentConfig, augment
from classify import preprocess_crop
from model import NUM_PINS, PinClassifier

HYPERPARAMS_PATH = Path("models/hyperparams.json")

DEFAULTS: dict[str, float | int] = {
    "lr": 3e-4,
    "weight_decay": 0.0,
    "dropout": 0.3,
    "batch_size": 16,
}

_TRAIN_AUGMENT = AugmentConfig(
    brightness_range=(-40, 40),
    noise_sigma_range=(3.0, 12.0),
    blur_kernels=(0, 0, 3, 3, 5),
    blur_sigma_range=(0.3, 1.8),
    max_rotation_deg=6.0,
    scale_range=(0.85, 1.15),
    grid_line_probability=0.4,
    grid_intensity_range=(90, 180),
)


def load_hyperparams() -> dict[str, float | int]:
    """Return training defaults, overridden by ``models/hyperparams.json``."""
    if HYPERPARAMS_PATH.exists():
        return {**DEFAULTS, **json.loads(HYPERPARAMS_PATH.read_text())}
    return dict(DEFAULTS)


class CropDataset(Dataset):
    """Pin-diagram crops with optional online augmentation."""

    def __init__(self, image_dir: Path, entries: list[tuple[str, list[int]]], *,
                 augment_cfg: AugmentConfig | None = None, seed: int = 0) -> None:
        self.image_dir, self.entries, self.augment_cfg = image_dir, entries, augment_cfg
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, pins = self.entries[idx]
        img = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing crop: {self.image_dir / filename}")
        if self.augment_cfg is not None:
            img = augment(img, self._rng, self.augment_cfg)
        return torch.from_numpy(preprocess_crop(img)).unsqueeze(0), torch.tensor(pins, dtype=torch.float32)


def split_entries(
    entries: list[tuple[str, list[int]]], val_fraction: float, seed: int,
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Shuffle and split into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(entries)).tolist()
    n_val = max(1, int(len(entries) * val_fraction))
    val_set = set(idx[:n_val])
    return [entries[i] for i in idx if i not in val_set], [entries[i] for i in idx if i in val_set]


def _train_loop(
    model: PinClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> tuple[float, float]:
    """Core loop. Restores best weights in-place. Returns ``(best_loss, best_acc)``."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss, best_acc = float("inf"), 0.0
    best_state: dict | None = None

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        if val_loader is not None:
            model.eval()
            val_loss = correct = total = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    val_loss += criterion(logits, labels).item() * images.size(0)
                    correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
                    total += labels.numel()
            val_loss /= len(val_loader.dataset)  # type: ignore[arg-type]
            metric, val_acc = val_loss, correct / total if total else 0.0
        else:
            metric, val_acc = epoch_loss, 0.0

        if metric < best_loss:
            best_loss, best_acc = metric, val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss, best_acc


def train_new_model(
    entries: list[tuple[str, list[int]]],
    crops_dir: Path,
    epochs: int,
    device: torch.device,
    seed: int,
    *,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    val_entries: list[tuple[str, list[int]]] | None = None,
) -> tuple[PinClassifier, float, float]:
    """Create, train, and return ``(model, best_loss, best_acc)``.

    Best weights are restored in-place before returning.
    """
    torch.manual_seed(seed)
    train_loader = DataLoader(
        CropDataset(crops_dir, entries, augment_cfg=_TRAIN_AUGMENT, seed=seed),
        batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_entries:
        val_loader = DataLoader(CropDataset(crops_dir, val_entries), batch_size=batch_size)
    model = PinClassifier(dropout=dropout).to(device)
    loss, acc = _train_loop(model, train_loader, val_loader,
                            epochs=epochs, device=device, lr=lr, weight_decay=weight_decay)
    return model, loss, acc
