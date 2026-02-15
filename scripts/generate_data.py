"""Generate synthetic pin diagram crops for CNN classifier training.

Renders all 512 possible 9-pin states as grayscale images that resemble
the real YOLO-cropped diagrams.

Usage:
    uv run python -m scripts.generate_data
    uv run python -m scripts.generate_data --train-per-pattern 30 --val-per-pattern 8
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path

import cv2
import numpy as np

from pinsheet_scanner.constants import NUM_PINS, PIN_POSITIONS

# Native rendering canvas size — close to real crop dimensions.
CANVAS_W, CANVAS_H = 50, 40

# Ellipse semi-axes at canvas resolution (calibrated against real crops).
KNOCKED_DOWN_AXES = (4, 1)  # wide horizontal bar
STANDING_AXES = (2, 2)  # compact circle


def _render_diagram(
    pins: list[int],
    *,
    bg: int = 200,
    ink: int = 40,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a clean pin diagram on a uniform background."""
    img = np.full((CANVAS_H, CANVAS_W), bg, dtype=np.uint8)

    for i, (nx, ny) in enumerate(PIN_POSITIONS):
        cx = int(nx * (CANVAS_W - 1))
        cy = int(ny * (CANVAS_H - 1))
        jitter = int(rng.integers(-1, 2)) if rng is not None else 0
        axes = KNOCKED_DOWN_AXES if pins[i] == 1 else STANDING_AXES
        ax = max(1 + pins[i], axes[0] + jitter)
        ay = max(1, axes[1] + jitter)
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, int(ink), -1)

    return img


def _random_grid_lines(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Optionally draw faint grid-line remnants at the edges."""
    if rng.random() < 0.4:
        return img

    h, w = img.shape
    intensity = int(rng.integers(100, 170))
    margin_y, margin_x = max(3, h // 8), max(3, w // 8)

    for y in [int(rng.integers(0, margin_y)), int(rng.integers(h - margin_y, h))]:
        if rng.random() < 0.5:
            cv2.line(img, (0, y), (w, y), intensity, 1)
    for x in [int(rng.integers(0, margin_x)), int(rng.integers(w - margin_x, w))]:
        if rng.random() < 0.5:
            cv2.line(img, (x, 0), (x, h), intensity, 1)

    return img


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations simulating real scan artifacts."""
    h, w = img.shape
    out = img.copy()

    out = _random_grid_lines(out, rng)

    # Brightness jitter
    out = np.clip(out.astype(np.int16) + int(rng.integers(-25, 26)), 0, 255).astype(
        np.uint8
    )

    # Gaussian noise
    noise = rng.normal(0, rng.uniform(2.0, 8.0), out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur
    ksize = int(rng.choice([0, 3, 3, 5]))
    if ksize > 0:
        out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=rng.uniform(0.4, 1.5))

    # Small rotation
    angle = rng.uniform(-4.0, 4.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderValue=int(out[0, 0]))

    # Aspect-ratio jitter
    new_w = max(10, int(w * rng.uniform(0.9, 1.1)))
    new_h = max(10, int(h * rng.uniform(0.9, 1.1)))
    out = cv2.resize(
        cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA),
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    return out


def generate_split(
    output_dir: Path,
    images_per_pattern: int,
    *,
    seed: int = 0,
) -> list[tuple[str, list[int]]]:
    """Generate one data split (train or val)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    entries: list[tuple[str, list[int]]] = []
    for pattern in itertools.product([0, 1], repeat=NUM_PINS):
        pins = list(pattern)
        tag = "".join(str(p) for p in pins)
        for v in range(images_per_pattern):
            img = _render_diagram(
                pins,
                bg=int(rng.integers(170, 220)),
                ink=int(rng.integers(20, 80)),
                rng=rng,
            )
            img = _augment(img, rng)
            filename = f"{tag}_{v:03d}.png"
            cv2.imwrite(str(output_dir / filename), img)
            entries.append((filename, pins))

    random.Random(seed).shuffle(entries)
    return entries


def write_labels(path: Path, entries: list[tuple[str, list[int]]]) -> None:
    """Write a CSV label file: filename, p0, p1, ..., p8."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + [f"p{i}" for i in range(NUM_PINS)])
        for filename, pins in entries:
            writer.writerow([filename] + pins)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic pin diagram training data."
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/classifier"), help="Root output dir."
    )
    p.add_argument(
        "--train-per-pattern",
        type=int,
        default=20,
        help="Augmented images per pattern for train split.",
    )
    p.add_argument(
        "--val-per-pattern",
        type=int,
        default=5,
        help="Augmented images per pattern for val split.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_patterns = 2**NUM_PINS

    print(f"Generating synthetic data in {args.output.resolve()}")
    print(f"  Patterns: {n_patterns}")
    print(
        f"  Train: {args.train_per_pattern}/pattern = {n_patterns * args.train_per_pattern}"
    )
    print(
        f"  Val:   {args.val_per_pattern}/pattern = {n_patterns * args.val_per_pattern}"
    )

    for split, per_pattern, seed_offset in [
        ("train", args.train_per_pattern, 0),
        ("val", args.val_per_pattern, 1),
    ]:
        print(f"\nGenerating {split} split...")
        entries = generate_split(
            args.output / split, per_pattern, seed=args.seed + seed_offset
        )
        write_labels(args.output / f"{split}_labels.csv", entries)
        print(f"  Wrote {len(entries)} images")

    print("\nDone.")


if __name__ == "__main__":
    main()
