"""Train the PinClassifier CNN on real labeled crops.

Reads ground-truth labels from ``debug_crops/labels.csv``, splits into
100 train / 20 val images, and trains with heavy online augmentation to
compensate for the small dataset.

Usage:
    uv run python -m scripts.train_classifier
    uv run python -m scripts.train_classifier --epochs 60 --lr 1e-3
    uv run python -m scripts.train_classifier --crops debug_crops/raw --labels debug_crops/labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pinsheet_scanner.augment import AugmentConfig, augment
from pinsheet_scanner.classify import preprocess_crop, resolve_device
from pinsheet_scanner.constants import NUM_PINS
from pinsheet_scanner.model import PinClassifier

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Heavier augmentation than the default — we only have ~100 training images.
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


def _load_labels(labels_csv: Path) -> list[tuple[str, list[int]]]:
    """Read ``(filename, pins)`` pairs from a CSV file."""
    entries: list[tuple[str, list[int]]] = []
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            pins = [int(row[f"p{i}"]) for i in range(NUM_PINS)]
            entries.append((row["filename"], pins))
    return entries


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


def _split_entries(
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
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: PinClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the PinClassifier CNN on real labeled crops."
    )
    p.add_argument(
        "--crops",
        type=Path,
        default=Path("debug_crops/raw"),
        help="Directory containing crop PNGs.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("debug_crops/labels.csv"),
        help="Ground-truth labels CSV.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/pin_classifier.pt"),
        help="Output weights path.",
    )
    p.add_argument(
        "--val-count",
        type=int,
        default=20,
        help="Number of images to hold out for validation.",
    )
    p.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--device", type=str, default=None, help="Device: cpu/cuda/mps.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(
            f"Labels not found at {args.labels}. Run `just label` first."
        )
    if not args.crops.exists():
        raise FileNotFoundError(
            f"Crops directory not found at {args.crops}. "
            "Run `just debug-crops <image>` first."
        )

    all_entries = _load_labels(args.labels)
    if len(all_entries) <= args.val_count:
        raise ValueError(
            f"Only {len(all_entries)} labeled images — need more than "
            f"--val-count={args.val_count} to have a training set."
        )

    train_entries, val_entries = _split_entries(all_entries, args.val_count, args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Total labeled: {len(all_entries)}")
    print(f"Train: {len(train_entries)}  |  Val: {len(val_entries)}")

    train_ds = RealCropDataset(
        args.crops, train_entries, augment_cfg=_TRAIN_AUGMENT, seed=args.seed
    )
    val_ds = RealCropDataset(args.crops, val_entries, augment_cfg=None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = PinClassifier().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
            torch.save(model.state_dict(), args.output)
            marker = " ✓"

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  {val_acc:9.2%}  {lr:10.1e}{marker}"
        )

    print(f"\nBest val loss: {best_val_loss:.4f}  |  Best val acc: {best_val_acc:.2%}")
    print(f"Saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
