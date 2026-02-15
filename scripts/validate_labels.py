"""Compare ground-truth labels against CNN predictions.

Reads labels from debug_crops/labels.csv, runs the CNN classifier on the
same crops, and reports per-pin and per-diagram accuracy.

Usage:
    uv run python -m scripts.validate_labels
    uv run python -m scripts.validate_labels --crops debug_crops/raw --labels debug_crops/labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from pinsheet_scanner.classify import (
    classify_pins_batch_with_confidence,
    load_classifier,
)
from pinsheet_scanner.constants import NUM_PINS
from pinsheet_scanner.pipeline import DEFAULT_CLASSIFIER_PATH


def _load_labels(labels_path: Path) -> dict[str, list[int]]:
    """Load ground-truth labels from CSV."""
    labels: dict[str, list[int]] = {}
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            pins = [int(row[f"p{i}"]) for i in range(NUM_PINS)]
            labels[row["filename"]] = pins
    return labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate CNN predictions against ground-truth labels."
    )
    p.add_argument(
        "--crops", type=Path, default=Path("debug_crops/raw"), help="Crops directory."
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("debug_crops/labels.csv"),
        help="Ground-truth labels CSV.",
    )
    p.add_argument(
        "--classifier-model",
        type=Path,
        default=None,
        help="CNN classifier weights (.pt).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    classifier_path: Path = args.classifier_model or DEFAULT_CLASSIFIER_PATH

    if not args.labels.exists():
        raise FileNotFoundError(
            f"Labels not found at {args.labels}. Run `just label` first."
        )
    if not args.crops.exists():
        raise FileNotFoundError(
            f"Crops directory not found at {args.crops}. "
            "Run `just debug-crops <image>` first."
        )

    labels = _load_labels(args.labels)
    if not labels:
        print("No labels found — nothing to validate.")
        return

    print(f"Loaded {len(labels)} ground-truth labels from {args.labels}")

    # Load crops in label order
    names = sorted(labels.keys())
    images = []
    valid_names = []
    for name in names:
        img = cv2.imread(str(args.crops / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: could not load {name}, skipping")
            continue
        images.append(img)
        valid_names.append(name)

    # Run CNN
    model, device = load_classifier(classifier_path)
    results = classify_pins_batch_with_confidence(model, images, device=device)

    # Compare
    total_pins = 0
    correct_pins = 0
    total_diagrams = 0
    correct_diagrams = 0
    per_pin_correct = [0] * NUM_PINS
    per_pin_total = [0] * NUM_PINS
    mismatches: list[tuple[str, list[int], list[int], float]] = []

    for name, (pred_pins, conf) in zip(valid_names, results):
        gt = labels[name]
        total_diagrams += 1
        diagram_ok = True

        for i in range(NUM_PINS):
            per_pin_total[i] += 1
            total_pins += 1
            if pred_pins[i] == gt[i]:
                correct_pins += 1
                per_pin_correct[i] += 1
            else:
                diagram_ok = False

        if diagram_ok:
            correct_diagrams += 1
        else:
            mismatches.append((name, gt, pred_pins, conf))

    # Report
    pin_acc = correct_pins / total_pins * 100 if total_pins else 0
    diag_acc = correct_diagrams / total_diagrams * 100 if total_diagrams else 0

    print(f"\n{'=' * 60}")
    print(f"  Results: {total_diagrams} diagrams, {total_pins} pins")
    print(f"{'=' * 60}")
    print(f"  Per-pin accuracy:    {correct_pins}/{total_pins} ({pin_acc:.1f}%)")
    print(
        f"  Per-diagram accuracy: {correct_diagrams}/{total_diagrams} ({diag_acc:.1f}%)"
    )

    print("\n  Per-position accuracy:")
    print(f"  {'Pin':>4}  {'Correct':>8}  {'Total':>6}  {'Acc':>7}")
    print(f"  {'-' * 30}")
    for i in range(NUM_PINS):
        acc = per_pin_correct[i] / per_pin_total[i] * 100 if per_pin_total[i] else 0
        marker = "" if acc >= 95 else " ←" if acc >= 80 else " ← LOW"
        print(
            f"  {i:>4}  {per_pin_correct[i]:>8}  {per_pin_total[i]:>6}  {acc:>6.1f}%{marker}"
        )

    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        print(f"  {'Name':<20} {'Ground Truth':>12} {'Prediction':>12} {'Conf':>6}")
        print(f"  {'-' * 54}")
        for name, gt, pred, conf in sorted(mismatches):
            gt_str = "".join(str(p) for p in gt)
            pred_str = "".join(str(p) for p in pred)
            diff = "".join("^" if g != p else " " for g, p in zip(gt, pred))
            print(f"  {name:<20} {gt_str:>12} {pred_str:>12} {conf:>5.0%}")
            print(f"  {'':20} {'':12} {diff:>12}")
    else:
        print("\n  No mismatches — perfect accuracy! 🎯")

    print()


if __name__ == "__main__":
    main()
