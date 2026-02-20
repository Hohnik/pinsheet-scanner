"""Training: dataset, training loop, and hyperparameter management."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from augment import AugmentConfig, augment
from classify import preprocess_crop
from model import NUM_PINS, PinClassifier

HYPERPARAMS_PATH = Path("models/hyperparams.json")

DEFAULTS: dict[str, float | int] = {
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "batch_size": 16,
}

LABEL_SMOOTH: float = 0.05   # prevents overconfidence on small datasets
WARMUP_FRACTION: float = 0.1  # first 10% of epochs = linear warmup

_TRAIN_AUGMENT = AugmentConfig(
    brightness_range=(-50, 50),
    noise_sigma_range=(2.0, 15.0),
    blur_kernels=(0, 0, 3, 3, 5),
    blur_sigma_range=(0.3, 1.8),
    max_rotation_deg=8.0,
    scale_range=(0.80, 1.20),
    grid_line_probability=0.4,
    grid_intensity_range=(90, 180),
    gamma_range=(0.5, 2.0),
)


def load_hyperparams() -> dict[str, float | int]:
    """Return training defaults, overridden by ``models/hyperparams.json``."""
    if HYPERPARAMS_PATH.exists():
        stored = json.loads(HYPERPARAMS_PATH.read_text())
        # filter to only known keys so stale keys (e.g. "scheduler") are ignored
        return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}
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


def _cosine_warmup_lambda(epoch: int, epochs: int, warmup: int) -> float:
    """LR multiplier: linear warmup then cosine decay to 1e-3 of peak."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 1e-3 + (1.0 - 1e-3) * 0.5 * (1 + math.cos(math.pi * progress))


def _train_loop(
    model: PinClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    *,
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    desc: str = "",
    leave: bool = True,
) -> tuple[float, float]:
    """Core loop. Restores best weights in-place. Returns ``(best_loss, best_acc)``."""
    from tqdm import tqdm

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = max(1, int(epochs * WARMUP_FRACTION))
    scheduler = LambdaLR(optimizer, lambda e: _cosine_warmup_lambda(e, epochs, warmup))

    best_loss, best_acc = float("inf"), 0.0
    best_state: dict | None = None

    bar = tqdm(range(epochs), desc=desc or "Training", unit="ep",
               leave=leave, dynamic_ncols=True) if desc else range(epochs)

    for _ in bar:
        model.train()
        epoch_loss = train_correct = train_total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Label smoothing: avoids overconfidence on small/noisy datasets
            labels_smooth = labels * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
            logits = model(images)
            loss = criterion(logits, labels_smooth)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            # Track train accuracy without an extra forward pass.
            with torch.no_grad():
                train_correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
                train_total   += labels.numel()
        epoch_loss /= len(train_loader.dataset)  # type: ignore[arg-type]
        epoch_train_acc = train_correct / train_total if train_total else 0.0

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
            # No val set (final retrain) — use training accuracy for display.
            metric, val_acc = epoch_loss, epoch_train_acc

        if metric < best_loss:
            best_loss, best_acc = metric, val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if desc and hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{metric:.4f}", acc=f"{val_acc:.1%}")  # type: ignore[union-attr]

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
    desc: str = "",
    leave: bool = True,
) -> tuple[PinClassifier, float, float]:
    """Create, train, and return ``(model, best_loss, best_acc)``.

    Pass ``desc`` to show a tqdm progress bar with that label.
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
                            epochs=epochs, device=device, lr=lr, weight_decay=weight_decay,
                            desc=desc, leave=leave)
    return model, loss, acc
