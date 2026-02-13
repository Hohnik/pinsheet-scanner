"""Train a YOLOv11n model to detect pin diagrams on bowling score sheets.

Usage:
    uv run python -m scripts.train
    uv run python -m scripts.train --data data/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model to detect pin diagrams."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dataset.yaml"),
        help="Path to the YOLO dataset YAML config (default: data/dataset.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained model to fine-tune from (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. -1 for auto-batch (default: -1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on: 'cpu', '0', 'mps', etc. None for auto-detect.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Project directory for saving results (default: runs).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pin_diagram",
        help="Experiment name (default: pin_diagram).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {args.data}. "
            "See data/dataset.yaml for the expected format and populate "
            "data/train/ and data/val/ with labeled images."
        )

    from ultralytics import YOLO

    # Load pretrained model (downloads automatically if not cached)
    model = YOLO(args.model)

    # Train
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Augmentation defaults that work well for document scans
        hsv_h=0.0,  # no hue shift (grayscale-ish documents)
        hsv_s=0.0,  # no saturation shift
        hsv_v=0.2,  # slight brightness variation
        degrees=5.0,  # small rotation (scans are mostly upright)
        translate=0.05,
        scale=0.2,
        flipud=0.0,  # no vertical flip (rows have a fixed order)
        fliplr=0.0,  # no horizontal flip (columns have a fixed order)
        mosaic=0.5,
    )

    # Validate on the val split
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Export best weights path
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nBest weights saved to: {best_weights}")
        print(f"Copy to models/ with:\n  cp {best_weights} models/pin_diagram.pt")
    else:
        print("\nTraining complete. Check runs/ for weights.")


if __name__ == "__main__":
    main()
