"""Tests for training utilities: scheduler factory, defaults, train loop."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from constants import NUM_PINS
from model import SpatialPinClassifier
from training import DEFAULTS, load_defaults, make_scheduler, train_and_evaluate


def _optimizer(lr=1e-3):
    return torch.optim.Adam(nn.Linear(4, 2).parameters(), lr=lr)


class TestMakeScheduler:
    @pytest.mark.parametrize("name", ["plateau", "cosine", "onecycle", "step"])
    def test_valid_names(self, name):
        assert make_scheduler(name, _optimizer(), 10, 5) is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            make_scheduler("nope", _optimizer(), 10, 5)


class TestLoadDefaults:
    def test_builtin_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("training.HYPERPARAMS_PATH", tmp_path / "x.json")
        assert load_defaults() == DEFAULTS

    def test_file_overrides(self, monkeypatch, tmp_path):
        import json

        p = tmp_path / "hp.json"
        p.write_text(json.dumps({"lr": 1e-2, "dropout": 0.1}))
        monkeypatch.setattr("training.HYPERPARAMS_PATH", p)
        d = load_defaults()
        assert d["lr"] == 1e-2 and d["dropout"] == 0.1


class TestSpatialPinClassifier:
    def test_output_shape(self):
        m = SpatialPinClassifier()
        assert m(torch.randn(2, 1, 64, 64)).shape == (2, NUM_PINS)

    @pytest.mark.parametrize("d", [0.0, 0.3, 0.5])
    def test_dropout_param(self, d):
        m = SpatialPinClassifier(dropout=d)
        assert m.head[0].p == pytest.approx(d)


class TestTrainAndEvaluate:
    @pytest.fixture()
    def tiny_dataset(self, tmp_path):
        import cv2
        import numpy as np

        d = tmp_path / "crops"
        d.mkdir()
        entries = []
        for i in range(6):
            name = f"crop_{i:02d}.png"
            cv2.imwrite(str(d / name), np.full((32, 32), 128 + i * 10, dtype=np.uint8))
            entries.append((name, [1 if (i + p) % 2 == 0 else 0 for p in range(NUM_PINS)]))
        return d, entries

    def test_returns_loss_and_acc(self, tiny_dataset):
        d, e = tiny_dataset
        vl, va = train_and_evaluate(e[:4], e[4:], d, 2, torch.device("cpu"), 0,
                                    lr=1e-3, weight_decay=0, dropout=0.3, batch_size=2, scheduler_name="plateau")
        assert vl >= 0 and 0 <= va <= 1

    @pytest.mark.parametrize("sched", ["plateau", "cosine", "onecycle", "step"])
    def test_all_schedulers(self, tiny_dataset, sched):
        d, e = tiny_dataset
        vl, va = train_and_evaluate(e[:4], e[4:], d, 2, torch.device("cpu"), 0,
                                    lr=1e-3, weight_decay=0, dropout=0.3, batch_size=4, scheduler_name=sched)
        assert vl >= 0
