"""Train a YOLOv11n model to detect pin diagrams on bowling score sheets.

Usage:
    uv run python -m scripts.train
    uv run python -m scripts.train --data data/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a YOLO model to detect pin diagrams."
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/dataset.yaml"),
        help="YOLO dataset YAML config.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained model to fine-tune from.",
    )
    p.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    p.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto).")
    p.add_argument(
        "--device", type=str, default=None, help="Device: 'cpu', '0', 'mps', etc."
    )
    p.add_argument(
        "--project", type=str, default="runs", help="Project directory for results."
    )
    p.add_argument("--name", type=str, default="pin_diagram", help="Experiment name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {args.data}. "
            "Populate data/train/ and data/val/ with labeled images first."
        )

    from ultralytics import YOLO  # type: ignore[attr-defined]

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
    )

    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nBest weights: {best_weights}")
        print(f"  cp {best_weights} models/pin_diagram.pt")
    else:
        print("\nTraining complete. Check runs/ for weights.")


if __name__ == "__main__":
    main()
