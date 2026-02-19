"""Shared training primitives for the SpatialPinClassifier CNN.

Provides the dataset, training loop, scheduler factory, hyperparameter
persistence, experiment logging, and train/val splitting used by the CLI
training commands (train, tune, kfold).
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
from dataclasses import dataclass
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

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

HYPERPARAMS_PATH = Path("models/hyperparams.json")
EXPERIMENTS_PATH = Path("experiments.jsonl")

DEFAULTS: dict[str, float | int | str] = {
    "lr": 3e-4,
    "weight_decay": 0.0,
    "dropout": 0.3,
    "batch_size": 16,
    "scheduler": "plateau",
}

# Heavier augmentation for training (small real dataset).
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
# Model bundle  (C2)
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def save_model_bundle(
    model: SpatialPinClassifier,
    output: Path,
    *,
    val_accuracy: float | None = None,
    extra: dict | None = None,
) -> None:
    """Save model weights and a JSON sidecar with training metadata.

    The sidecar is written to ``output.with_suffix('.json')`` and is read
    by :func:`~classify.load_classifier` to determine the architecture.

    Args:
        model: Trained model to checkpoint.
        output: Destination ``.pt`` file path.
        val_accuracy: Best validation accuracy to record in the sidecar.
        extra: Any additional key-value pairs to include in the sidecar.
    """
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


# ---------------------------------------------------------------------------
# Experiment log  (C3)
# ---------------------------------------------------------------------------


def log_experiment(record: dict) -> None:
    """Append a training record to ``experiments.jsonl``.

    Automatically adds ``timestamp`` and ``git_sha`` fields.
    """
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        **record,
    }
    with open(EXPERIMENTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


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
    if scheduler_name == "onecycle":
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)  # type: ignore[arg-type]
    else:
        scheduler.step()


# ---------------------------------------------------------------------------
# Training setup — shared by train_and_evaluate / retrain_all
# ---------------------------------------------------------------------------


@dataclass
class TrainingComponents:
    """All objects needed for a training run, built from common parameters."""

    model: SpatialPinClassifier
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: LRScheduler
    scheduler_name: str
    step_per_batch: bool


def build_training_components(
    device: torch.device,
    seed: int,
    *,
    lr: float,
    weight_decay: float,
    dropout: float,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
) -> TrainingComponents:
    """Construct model, criterion, optimizer, and scheduler."""
    torch.manual_seed(seed)
    model = SpatialPinClassifier(dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch)
    return TrainingComponents(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_name=scheduler_name,
        step_per_batch=scheduler_name == "onecycle",
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: SpatialPinClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scheduler: LRScheduler | None = None,
) -> float:
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
    model: SpatialPinClassifier,
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
    """Train a ``SpatialPinClassifier`` and return ``(best_val_loss, best_val_acc)``."""
    train_ds = RealCropDataset(
        crops_dir, train_entries, augment_cfg=TRAIN_AUGMENT, seed=seed
    )
    val_ds = RealCropDataset(crops_dir, val_entries, augment_cfg=None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    tc = build_training_components(
        device, seed,
        lr=lr, weight_decay=weight_decay, dropout=dropout,
        scheduler_name=scheduler_name, epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for _epoch in range(1, epochs + 1):
        train_one_epoch(
            tc.model, train_loader, tc.criterion, tc.optimizer, device,
            scheduler=tc.scheduler if tc.step_per_batch else None,
        )
        val_loss, val_acc = evaluate(tc.model, val_loader, tc.criterion, device)
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
        step_scheduler(tc.scheduler, tc.scheduler_name, metric=val_loss)

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
    val_accuracy: float | None = None,
    extra_bundle: dict | None = None,
) -> float:
    """Train on *all* entries, save best checkpoint with bundle. Returns best loss."""
    ds = RealCropDataset(crops_dir, entries, augment_cfg=TRAIN_AUGMENT, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    tc = build_training_components(
        device, seed,
        lr=lr, weight_decay=weight_decay, dropout=dropout,
        scheduler_name=scheduler_name, epochs=epochs,
        steps_per_epoch=len(loader),
    )

    best_loss = float("inf")
    for _epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            tc.model, loader, tc.criterion, tc.optimizer, device,
            scheduler=tc.scheduler if tc.step_per_batch else None,
        )
        if loss < best_loss:
            best_loss = loss
            save_model_bundle(
                tc.model, output,
                val_accuracy=val_accuracy,
                extra=extra_bundle,
            )
        step_scheduler(tc.scheduler, tc.scheduler_name, metric=loss)

    return best_loss


# ---------------------------------------------------------------------------
# K-fold cross-validation + retrain
# ---------------------------------------------------------------------------


@dataclass
class KFoldResult:
    """Summary of a K-fold cross-validation run."""

    losses: list[float]
    accuracies: list[float]
    mean_loss: float
    mean_acc: float
    std_acc: float
    std_loss: float
    final_train_loss: float
    output: Path


def kfold_train(
    all_entries: list[tuple[str, list[int]]],
    crops_dir: Path,
    output: Path,
    *,
    folds: int,
    epochs: int,
    device: torch.device,
    seed: int = 42,
    hp_kwargs: dict,
) -> KFoldResult:
    """Run K-fold cross-validation, retrain on all data, and save the model.

    This is the core logic extracted from the ``train`` CLI command so it
    can be tested and reused programmatically.

    Args:
        all_entries: Full labeled dataset as ``(filename, pins)`` pairs.
        crops_dir: Directory containing crop images.
        output: Where to save the final ``.pt`` weights.
        folds: Number of cross-validation folds.
        epochs: Training epochs per fold (and final retrain).
        device: Torch device.
        seed: Random seed for reproducibility.
        hp_kwargs: Hyperparameter dict with keys ``lr``, ``weight_decay``,
            ``dropout``, ``batch_size``, ``scheduler_name``.

    Returns:
        :class:`KFoldResult` with per-fold and aggregate metrics.
    """
    from sklearn.model_selection import KFold

    logger = logging.getLogger(__name__)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    losses: list[float] = []
    accs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_entries)):
        train_e = [all_entries[i] for i in train_idx]
        val_e = [all_entries[i] for i in val_idx]
        logger.info(
            "Fold %d/%d  (train=%d, val=%d)", fold + 1, folds, len(train_e), len(val_e),
        )
        val_loss, val_acc = train_and_evaluate(
            train_entries=train_e, val_entries=val_e,
            crops_dir=crops_dir, epochs=epochs, device=device,
            seed=seed + fold, **hp_kwargs,
        )
        losses.append(val_loss)
        accs.append(val_acc)
        logger.info("  → loss=%.4f  acc=%.2f%%", val_loss, val_acc * 100)

    mean_loss = float(np.mean(losses))
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    std_loss = float(np.std(losses))

    logger.info("Retraining on all %d images for %d epochs...", len(all_entries), epochs)
    extra_bundle = {
        "folds": folds, "epochs": epochs,
        **{k: v for k, v in hp_kwargs.items() if k != "scheduler_name"},
    }
    best_loss = retrain_all(
        all_entries, crops_dir, output, device, epochs, seed,
        val_accuracy=mean_acc, extra_bundle=extra_bundle,
        **hp_kwargs,
    )

    return KFoldResult(
        losses=losses,
        accuracies=accs,
        mean_loss=mean_loss,
        mean_acc=mean_acc,
        std_acc=std_acc,
        std_loss=std_loss,
        final_train_loss=best_loss,
        output=output,
    )
