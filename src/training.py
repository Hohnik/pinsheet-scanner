"""Training primitives for the SpatialPinClassifier CNN.

Provides dataset, training loop, scheduler factory, hyperparameter
persistence, experiment logging, and model bundling.
"""

from __future__ import annotations

import datetime
import json
import subprocess
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

from augment import AugmentConfig, augment
from classify import preprocess_crop
from constants import NUM_PINS
from model import SpatialPinClassifier

HYPERPARAMS_PATH = Path("models/hyperparams.json")
EXPERIMENTS_PATH = Path("experiments.jsonl")

DEFAULTS: dict[str, float | int | str] = {
    "lr": 3e-4,
    "weight_decay": 0.0,
    "dropout": 0.3,
    "batch_size": 16,
    "scheduler": "plateau",
}

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


def load_defaults() -> dict[str, float | int | str]:
    """Return training defaults, preferring ``models/hyperparams.json``."""
    if HYPERPARAMS_PATH.exists():
        return {**DEFAULTS, **json.loads(HYPERPARAMS_PATH.read_text())}
    return dict(DEFAULTS)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_model_bundle(
    model: SpatialPinClassifier, output: Path, *,
    val_accuracy: float | None = None, extra: dict | None = None,
) -> None:
    """Save weights + JSON sidecar with architecture and metadata."""
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    bundle: dict = {
        "architecture": type(model).__name__,
        "input_size": [1, 64, 64],
        "num_pins": NUM_PINS,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    if val_accuracy is not None:
        bundle["val_accuracy"] = round(val_accuracy, 6)
    if extra:
        bundle.update(extra)
    output.with_suffix(".json").write_text(json.dumps(bundle, indent=2) + "\n")


def log_experiment(record: dict) -> None:
    """Append a timestamped record to ``experiments.jsonl``."""
    record = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
              "git_sha": _git_sha(), **record}
    with open(EXPERIMENTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


class RealCropDataset(Dataset):
    """Loads real crop images with optional online augmentation."""

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
            raise FileNotFoundError(f"Could not load {self.image_dir / filename}")
        if self.augment_cfg is not None:
            img = augment(img, self._rng, self.augment_cfg)
        return torch.from_numpy(preprocess_crop(img)).unsqueeze(0), torch.tensor(pins, dtype=torch.float32)


def split_entries(
    entries: list[tuple[str, list[int]]], val_count: int, seed: int,
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Shuffle and split into (train, val)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(entries)).tolist()
    val_set = set(idx[:val_count])
    return [entries[i] for i in idx if i not in val_set], [entries[i] for i in idx if i in val_set]


def make_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int) -> LRScheduler:
    match name:
        case "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        case "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        case "onecycle":
            return OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]["lr"] * 10,
                              epochs=epochs, steps_per_epoch=steps_per_epoch)
        case "step":
            return StepLR(optimizer, step_size=15, gamma=0.5)
        case _:
            raise ValueError(f"Unknown scheduler: {name!r}")


def _step_scheduler(scheduler: LRScheduler, name: str, metric: float | None = None) -> None:
    if name == "onecycle":
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)  # type: ignore[arg-type]
    else:
        scheduler.step()


def _run_training_loop(
    model: SpatialPinClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    scheduler_name: str,
    save_path: Path | None = None,
    val_accuracy: float | None = None,
    extra_bundle: dict | None = None,
) -> tuple[float, float]:
    """Core training loop.  Returns ``(best_loss, best_acc)``."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(scheduler_name, optimizer, epochs, len(train_loader))
    step_per_batch = scheduler_name == "onecycle"

    best_loss, best_acc = float("inf"), 0.0

    for _ in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_per_batch:
                scheduler.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        # Evaluate
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
            n = len(val_loader.dataset)  # type: ignore[arg-type]
            val_loss /= n
            val_acc = correct / total if total else 0.0
            metric = val_loss
        else:
            metric = epoch_loss
            val_loss, val_acc = epoch_loss, 0.0

        if metric < best_loss:
            best_loss, best_acc = metric, val_acc
            if save_path is not None:
                save_model_bundle(model, save_path, val_accuracy=val_accuracy, extra=extra_bundle)

        _step_scheduler(scheduler, scheduler_name, metric=metric)

    return best_loss, best_acc


def train_and_evaluate(
    train_entries: list[tuple[str, list[int]]],
    val_entries: list[tuple[str, list[int]]],
    crops_dir: Path, epochs: int, device: torch.device, seed: int, *,
    lr: float, weight_decay: float, dropout: float, batch_size: int, scheduler_name: str,
) -> tuple[float, float]:
    """Train a SpatialPinClassifier and return ``(best_val_loss, best_val_acc)``."""
    torch.manual_seed(seed)
    train_loader = DataLoader(
        RealCropDataset(crops_dir, train_entries, augment_cfg=TRAIN_AUGMENT, seed=seed),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        RealCropDataset(crops_dir, val_entries),
        batch_size=batch_size)
    model = SpatialPinClassifier(dropout=dropout).to(device)
    return _run_training_loop(
        model, train_loader, val_loader, epochs=epochs, device=device,
        lr=lr, weight_decay=weight_decay, scheduler_name=scheduler_name)


def retrain_all(
    entries: list[tuple[str, list[int]]], crops_dir: Path, output: Path,
    device: torch.device, epochs: int, seed: int, *,
    lr: float, weight_decay: float, dropout: float, batch_size: int, scheduler_name: str,
    val_accuracy: float | None = None, extra_bundle: dict | None = None,
) -> float:
    """Train on all entries, save best checkpoint.  Returns best loss."""
    torch.manual_seed(seed)
    loader = DataLoader(
        RealCropDataset(crops_dir, entries, augment_cfg=TRAIN_AUGMENT, seed=seed),
        batch_size=batch_size, shuffle=True)
    model = SpatialPinClassifier(dropout=dropout).to(device)
    best_loss, _ = _run_training_loop(
        model, loader, None, epochs=epochs, device=device,
        lr=lr, weight_decay=weight_decay, scheduler_name=scheduler_name,
        save_path=output, val_accuracy=val_accuracy, extra_bundle=extra_bundle)
    return best_loss
