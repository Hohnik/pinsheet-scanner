"""Shared I/O helpers for ground-truth pin-state label CSV files.

The canonical CSV format has columns ``filename, p0, p1, …, p8`` where
each ``pN`` is ``0`` (standing) or ``1`` (knocked down).

This module provides a single source of truth for reading and writing
that format, used by the ``train-classifier``, ``accuracy``, and
``label`` CLI commands.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .constants import NUM_PINS

# Column names: filename, p0, p1, …, p8
_PIN_COLUMNS = [f"p{i}" for i in range(NUM_PINS)]
_HEADER = ["filename"] + _PIN_COLUMNS


def load_labels_as_dict(labels_path: Path) -> dict[str, list[int]]:
    """Read labels from *labels_path* into ``{filename: [p0, …, p8]}``.

    Returns an empty dict if the file does not exist.

    Raises:
        FileNotFoundError: Only when the caller needs to distinguish
            "missing file" from "empty file" — this function silently
            returns ``{}`` for a missing path so that callers who treat
            absence as "nothing labeled yet" don't need extra checks.
    """
    if not labels_path.exists():
        return {}

    labels: dict[str, list[int]] = {}
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            pins = [int(row[col]) for col in _PIN_COLUMNS]
            labels[row["filename"]] = pins
    return labels


def load_labels_as_list(labels_path: Path) -> list[tuple[str, list[int]]]:
    """Read labels from *labels_path* into ``[(filename, [p0, …, p8]), …]``.

    Unlike :func:`load_labels_as_dict`, this preserves the CSV row order
    and is the natural format for building a training dataset.

    Raises:
        FileNotFoundError: If *labels_path* does not exist.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    entries: list[tuple[str, list[int]]] = []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            pins = [int(row[col]) for col in _PIN_COLUMNS]
            entries.append((row["filename"], pins))
    return entries


def save_labels(labels_path: Path, labels: dict[str, list[int]]) -> None:
    """Atomically write all *labels* to *labels_path* in CSV format.

    Writes to a temporary file first, then renames it over the target so
    that a crash mid-write never leaves a truncated file.
    """
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = labels_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for filename in sorted(labels):
            writer.writerow([filename] + labels[filename])
    tmp.replace(labels_path)
