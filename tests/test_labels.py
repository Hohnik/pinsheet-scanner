"""Tests for label CSV I/O."""

from pathlib import Path

import pytest

from constants import NUM_PINS
from labels import load_labels_as_dict, load_labels_as_list, save_labels

_HEADER = "filename," + ",".join(f"p{i}" for i in range(NUM_PINS))
_ROWS = [
    ("c00_r00.png", [1, 1, 0, 0, 1, 0, 0, 0, 0]),
    ("c00_r01.png", [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("c01_r00.png", [1, 1, 1, 1, 1, 1, 1, 1, 1]),
]


def _write_csv(path, rows=None):
    rows = rows or _ROWS
    lines = [_HEADER] + [f"{n},{','.join(str(p) for p in pins)}" for n, pins in rows]
    path.write_text("\n".join(lines) + "\n")


class TestLoad:
    def test_dict_roundtrip(self, tmp_path):
        p = tmp_path / "l.csv"
        _write_csv(p)
        d = load_labels_as_dict(p)
        assert len(d) == len(_ROWS) and all(d[n] == pins for n, pins in _ROWS)

    def test_list_preserves_order(self, tmp_path):
        p = tmp_path / "l.csv"
        _write_csv(p)
        entries = load_labels_as_list(p)
        assert [(n, pins) for n, pins in entries] == _ROWS

    def test_missing_dict_returns_empty(self, tmp_path):
        assert load_labels_as_dict(tmp_path / "x.csv") == {}

    def test_missing_list_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_labels_as_list(tmp_path / "x.csv")


class TestSave:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "l.csv"
        original = {n: pins for n, pins in _ROWS}
        save_labels(p, original)
        assert load_labels_as_dict(p) == original

    def test_sorted_filenames(self, tmp_path):
        p = tmp_path / "l.csv"
        save_labels(p, {"z.png": [1] * NUM_PINS, "a.png": [0] * NUM_PINS})
        entries = load_labels_as_list(p)
        assert entries[0][0] == "a.png"

    def test_overwrites(self, tmp_path):
        p = tmp_path / "l.csv"
        save_labels(p, {"old.png": [0] * NUM_PINS})
        save_labels(p, {"new.png": [1] * NUM_PINS})
        assert "old.png" not in load_labels_as_dict(p) and "new.png" in load_labels_as_dict(p)

    def test_creates_parents(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "l.csv"
        save_labels(p, {"a.png": [0] * NUM_PINS})
        assert p.exists()
