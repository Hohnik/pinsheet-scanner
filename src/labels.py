"""Label CSV I/O for ground-truth pin states.

CSV format: ``filename, p0, p1, …, p8`` where each ``pN`` is 0 or 1.
"""

from __future__ import annotations

import csv
from pathlib import Path

_PIN_COLUMNS = [f"p{i}" for i in range(9)]
_HEADER = ["filename"] + _PIN_COLUMNS


def load_labels_as_dict(labels_path: Path) -> dict[str, list[int]]:
    """Read labels into ``{filename: [p0, …, p8]}``.  Returns ``{}`` if missing."""
    if not labels_path.exists():
        return {}
    labels: dict[str, list[int]] = {}
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["filename"]] = [int(row[c]) for c in _PIN_COLUMNS]
    return labels


def load_labels_as_list(labels_path: Path) -> list[tuple[str, list[int]]]:
    """Read labels into ``[(filename, [p0, …, p8]), …]``, preserving order."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    entries: list[tuple[str, list[int]]] = []
    with open(labels_path, newline="") as f:
        for row in csv.DictReader(f):
            entries.append((row["filename"], [int(row[c]) for c in _PIN_COLUMNS]))
    return entries


def save_labels(labels_path: Path, labels: dict[str, list[int]]) -> None:
    """Atomically write labels (via temp file + rename)."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = labels_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for filename in sorted(labels):
            writer.writerow([filename] + labels[filename])
    tmp.replace(labels_path)
