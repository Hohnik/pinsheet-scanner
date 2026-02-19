"""Shared training primitives for the PinClassifier CNN.

Provides the dataset, training loop, scheduler factory, hyperparameter
persistence, and train/val splitting used by the CLI training commands
(train-classifier, tune, kfold).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset

from .augment import AugmentConfig, augment
from .classify import preprocess_crop
from .model import PinClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HYPERPARAMS_PATH = Path("models/hyperparams.json")

DEFAULTS: dict[str, float | int | str] = {
    "lr": 3e-4,
    "weight_decay": 0.0,
    "dropout": 0.3,
    "batch_size": 16,
    "scheduler": "plateau",
}

# Heavier augmentation — we only have ~100 training images.
TRAIN_AUGMENT = AugmentConfig(
    brightness_range=(-40, 40),
    noise_sigma_range=(3.0, 12.0),
    blur_kernels=(0, 0, 3, 3, 5),
    blur_sigma_range=(0.3, 1.8),
    max_rotation_deg=6.0,
    scale_range=(0.85, 1.15),
    grid_line_probability=0.4,
    grid_intensity_range=(90, 180),
)


# ---------------------------------------------------------------------------
# Hyperparameter persistence
# ---------------------------------------------------------------------------


def load_defaults() -> dict[str, float | int | str]:
    """Return training defaults, preferring ``models/hyperparams.json``."""
    if HYPERPARAMS_PATH.exists():
        saved = json.loads(HYPERPARAMS_PATH.read_text())
        return {**DEFAULTS, **saved}
    return dict(DEFAULTS)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RealCropDataset(Dataset):
    """Loads real crop images with optional online augmentation."""

    def __init__(
        self,
        image_dir: Path,
        entries: list[tuple[str, list[int]]],
        *,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.image_dir = image_dir
        self.entries = entries
        self.augment_cfg = augment_cfg
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, pins = self.entries[idx]
        img = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load {self.image_dir / filename}")
        if self.augment_cfg is not None:
            img = augment(img, self._rng, self.augment_cfg)
        processed = preprocess_crop(img)
        tensor = torch.from_numpy(processed).unsqueeze(0)  # (1, H, W)
        return tensor, torch.tensor(pins, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------


def split_entries(
    entries: list[tuple[str, list[int]]],
    val_count: int,
    seed: int,
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Shuffle and split into (train, val)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(entries)).tolist()
    val_indices = set(indices[:val_count])
    train = [entries[i] for i in indices if i not in val_indices]
    val = [entries[i] for i in indices if i in val_indices]
    return train, val


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def make_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
) -> LRScheduler:
    """Create a learning-rate scheduler by name."""
    match name:
        case "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        case "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        case "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]["lr"] * 10,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
            )
        case "step":
            return StepLR(optimizer, step_size=15, gamma=0.5)
        case _:
            raise ValueError(f"Unknown scheduler: {name!r}")


def step_scheduler(
    scheduler: LRScheduler,
    scheduler_name: str,
    metric: float | None = None,
) -> None:
    """Step a per-epoch scheduler (no-op for onecycle which steps per-batch)."""
    if scheduler_name == "onecycle":
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)  # type: ignore[arg-type]
    else:
        scheduler.step()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: PinClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scheduler: LRScheduler | None = None,
) -> float:
    """Train for one epoch, optionally stepping a per-batch scheduler."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: PinClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(loss, per-pin accuracy)``."""
    model.eval()
    total_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        total += labels.numel()
    n = len(loader.dataset)  # type: ignore[arg-type]
    return total_loss / n, correct / total if total else 0.0


def train_and_evaluate(
    train_entries: list[tuple[str, list[int]]],
    val_entries: list[tuple[str, list[int]]],
    crops_dir: Path,
    epochs: int,
    device: torch.device,
    seed: int,
    *,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    scheduler_name: str,
) -> tuple[float, float]:
    """Train a model and return ``(best_val_loss, best_val_acc)``.

    Reusable core used by train-classifier, tune, and kfold commands.
    """
    torch.manual_seed(seed)

    train_ds = RealCropDataset(
        crops_dir, train_entries, augment_cfg=TRAIN_AUGMENT, seed=seed
    )
    val_ds = RealCropDataset(crops_dir, val_entries, augment_cfg=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PinClassifier(dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = make_scheduler(scheduler_name, optimizer, epochs, len(train_loader))
    step_per_batch = scheduler_name == "onecycle"

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for _epoch in range(1, epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler if step_per_batch else None,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

        step_scheduler(scheduler, scheduler_name, metric=val_loss)

    return best_val_loss, best_val_acc


def retrain_all(
    entries: list[tuple[str, list[int]]],
    crops_dir: Path,
    output: Path,
    device: torch.device,
    epochs: int,
    seed: int,
    *,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    scheduler_name: str,
) -> float:
    """Train on *all* entries (no validation) and save the best checkpoint.

    Returns the best training loss.
    """
    torch.manual_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    ds = RealCropDataset(crops_dir, entries, augment_cfg=TRAIN_AUGMENT, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PinClassifier(dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = make_scheduler(scheduler_name, optimizer, epochs, len(loader))
    step_per_batch = scheduler_name == "onecycle"

    best_loss = float("inf")
    for _epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler if step_per_batch else None,
        )
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), output)
        step_scheduler(scheduler, scheduler_name, metric=loss)

    return best_loss
