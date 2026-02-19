"""Tests for the shared label CSV I/O module."""

from pathlib import Path

import pytest

from constants import NUM_PINS
from labels import (
    load_labels_as_dict,
    load_labels_as_list,
    save_labels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER = "filename," + ",".join(f"p{i}" for i in range(NUM_PINS))

_ROWS = [
    ("c00_r00.png", [1, 1, 0, 0, 1, 0, 0, 0, 0]),
    ("c00_r01.png", [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("c01_r00.png", [1, 1, 1, 1, 1, 1, 1, 1, 1]),
]


def _write_csv(path: Path, rows: list[tuple[str, list[int]]] | None = None) -> None:
    """Write a well-formed labels CSV to *path*."""
    rows = rows if rows is not None else _ROWS
    lines = [_HEADER]
    for filename, pins in rows:
        lines.append(filename + "," + ",".join(str(p) for p in pins))
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# load_labels_as_dict
# ---------------------------------------------------------------------------


class TestLoadLabelsAsDict:
    def test_returns_dict_with_correct_keys(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        labels = load_labels_as_dict(csv_path)
        assert set(labels.keys()) == {name for name, _ in _ROWS}

    def test_returns_correct_pin_values(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        labels = load_labels_as_dict(csv_path)
        for name, pins in _ROWS:
            assert labels[name] == pins

    def test_each_entry_has_num_pins_values(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        labels = load_labels_as_dict(csv_path)
        for pins in labels.values():
            assert len(pins) == NUM_PINS

    def test_values_are_ints(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        labels = load_labels_as_dict(csv_path)
        for pins in labels.values():
            assert all(isinstance(p, int) for p in pins)

    def test_missing_file_returns_empty_dict(self, tmp_path: Path):
        labels = load_labels_as_dict(tmp_path / "nonexistent.csv")
        assert labels == {}

    def test_empty_csv_returns_empty_dict(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text(_HEADER + "\n")
        labels = load_labels_as_dict(csv_path)
        assert labels == {}

    def test_single_row(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [_ROWS[0]])
        labels = load_labels_as_dict(csv_path)
        assert len(labels) == 1
        assert labels[_ROWS[0][0]] == _ROWS[0][1]


# ---------------------------------------------------------------------------
# load_labels_as_list
# ---------------------------------------------------------------------------


class TestLoadLabelsAsList:
    def test_returns_list_of_tuples(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        entries = load_labels_as_list(csv_path)
        assert isinstance(entries, list)
        assert all(isinstance(e, tuple) and len(e) == 2 for e in entries)

    def test_preserves_csv_row_order(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        entries = load_labels_as_list(csv_path)
        assert [name for name, _ in entries] == [name for name, _ in _ROWS]

    def test_correct_pin_values(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        entries = load_labels_as_list(csv_path)
        for (name_got, pins_got), (name_exp, pins_exp) in zip(entries, _ROWS):
            assert name_got == name_exp
            assert pins_got == pins_exp

    def test_each_entry_has_num_pins_values(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)
        for _, pins in load_labels_as_list(csv_path):
            assert len(pins) == NUM_PINS

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_labels_as_list(tmp_path / "nonexistent.csv")

    def test_empty_csv_returns_empty_list(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text(_HEADER + "\n")
        assert load_labels_as_list(csv_path) == []


# ---------------------------------------------------------------------------
# save_labels
# ---------------------------------------------------------------------------


class TestSaveLabels:
    def test_creates_file(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        labels = {name: pins for name, pins in _ROWS}
        save_labels(csv_path, labels)
        assert csv_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        csv_path = tmp_path / "sub" / "dir" / "labels.csv"
        save_labels(csv_path, {"a.png": [0] * NUM_PINS})
        assert csv_path.exists()

    def test_roundtrip_as_dict(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        original = {name: pins for name, pins in _ROWS}
        save_labels(csv_path, original)
        loaded = load_labels_as_dict(csv_path)
        assert loaded == original

    def test_roundtrip_as_list(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        original = {name: pins for name, pins in _ROWS}
        save_labels(csv_path, original)
        loaded = load_labels_as_list(csv_path)
        # save_labels sorts by filename, so the list should be sorted
        expected = sorted(_ROWS, key=lambda t: t[0])
        assert loaded == expected

    def test_filenames_are_sorted(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        labels = {"z.png": [1] * NUM_PINS, "a.png": [0] * NUM_PINS}
        save_labels(csv_path, labels)
        entries = load_labels_as_list(csv_path)
        assert entries[0][0] == "a.png"
        assert entries[1][0] == "z.png"

    def test_overwrites_existing_file(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        save_labels(csv_path, {"old.png": [0] * NUM_PINS})
        save_labels(csv_path, {"new.png": [1] * NUM_PINS})
        loaded = load_labels_as_dict(csv_path)
        assert "old.png" not in loaded
        assert "new.png" in loaded

    def test_empty_dict_writes_header_only(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        save_labels(csv_path, {})
        loaded = load_labels_as_dict(csv_path)
        assert loaded == {}
        # File should exist and have the header line
        content = csv_path.read_text()
        assert content.startswith("filename,")

    def test_no_temp_file_left_behind(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        save_labels(csv_path, {"a.png": [0] * NUM_PINS})
        tmp_file = csv_path.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_header_contains_all_pin_columns(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        save_labels(csv_path, {"a.png": [0] * NUM_PINS})
        first_line = csv_path.read_text().splitlines()[0]
        columns = first_line.split(",")
        assert columns[0] == "filename"
        for i in range(NUM_PINS):
            assert columns[i + 1] == f"p{i}"


# ---------------------------------------------------------------------------
# Cross-format consistency
# ---------------------------------------------------------------------------


class TestCrossFormatConsistency:
    """Verify that dict and list loaders agree on the same file."""

    def test_dict_and_list_contain_same_data(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path)

        as_dict = load_labels_as_dict(csv_path)
        as_list = load_labels_as_list(csv_path)

        assert len(as_dict) == len(as_list)
        for name, pins in as_list:
            assert as_dict[name] == pins

    def test_save_then_load_both_formats(self, tmp_path: Path):
        csv_path = tmp_path / "labels.csv"
        original = {"b.png": [1, 0, 1, 0, 1, 0, 1, 0, 1], "a.png": [0] * NUM_PINS}
        save_labels(csv_path, original)

        as_dict = load_labels_as_dict(csv_path)
        as_list = load_labels_as_list(csv_path)

        assert as_dict == original
        for name, pins in as_list:
            assert original[name] == pins
