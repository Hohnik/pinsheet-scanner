"""Train the PinClassifier CNN on synthetic pin diagram data.

Usage:
    uv run python -m scripts.train_classifier
    uv run python -m scripts.train_classifier --epochs 30 --lr 1e-3
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

from pinsheet_scanner.classify import _resolve_device
from pinsheet_scanner.constants import CLASSIFIER_INPUT_SIZE, NUM_PINS
from pinsheet_scanner.model import PinClassifier

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PinDiagramDataset(Dataset):
    """Loads synthetic pin diagram images + CSV labels for training."""

    def __init__(self, image_dir: Path, labels_csv: Path) -> None:
        self.image_dir = image_dir
        self.entries: list[tuple[str, list[int]]] = []

        with open(labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                pins = [int(row[f"p{i}"]) for i in range(NUM_PINS)]
                self.entries.append((row["filename"], pins))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, pins = self.entries[idx]
        img = cv2.imread(str(self.image_dir / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load {self.image_dir / filename}")

        w, h = CLASSIFIER_INPUT_SIZE
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, torch.tensor(pins, dtype=torch.float32)


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
    """Return (loss, per-pin accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader.dataset), correct / total if total else 0.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PinClassifier CNN.")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/classifier"),
        help="Data root directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/pin_classifier.pt"),
        help="Output weights path.",
    )
    p.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    p.add_argument("--device", type=str, default=None, help="Device: cpu/cuda/mps.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root: Path = args.data

    required = [
        data_root / "train_labels.csv",
        data_root / "val_labels.csv",
        data_root / "train",
        data_root / "val",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `uv run python -m scripts.generate_data` first."
            )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    train_ds = PinDiagramDataset(data_root / "train", data_root / "train_labels.csv")
    val_ds = PinDiagramDataset(data_root / "val", data_root / "val_labels.csv")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    model = PinClassifier().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}"
    )
    print("-" * 55)

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
