"""Tests for training utilities and PinClassifier model."""

from pathlib import Path

import pytest
import torch

from model import NUM_PINS, PinClassifier
from training import DEFAULTS, load_hyperparams, train_new_model


class TestLoadHyperparams:
    def test_builtin_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("training.HYPERPARAMS_PATH", tmp_path / "x.json")
        assert load_hyperparams() == DEFAULTS

    def test_file_overrides(self, monkeypatch, tmp_path):
        import json

        p = tmp_path / "hp.json"
        p.write_text(json.dumps({"lr": 1e-2, "dropout": 0.1}))
        monkeypatch.setattr("training.HYPERPARAMS_PATH", p)
        d = load_hyperparams()
        assert d["lr"] == 1e-2 and d["dropout"] == 0.1


class TestPinClassifier:
    def test_output_shape(self):
        m = PinClassifier()
        assert m(torch.randn(2, 1, 64, 64)).shape == (2, NUM_PINS)

    @pytest.mark.parametrize("d", [0.0, 0.3, 0.5])
    def test_dropout_param(self, d):
        m = PinClassifier(dropout=d)
        assert m.head[0].p == pytest.approx(d)

    def test_eval_deterministic(self):
        m = PinClassifier()
        m.eval()
        x = torch.randn(1, 1, 64, 64)
        torch.testing.assert_close(m(x), m(x))


class TestTrainNewModel:
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

    def test_returns_model_and_metrics(self, tiny_dataset):
        d, e = tiny_dataset
        model, vl, va = train_new_model(e[:4], d, 2, torch.device("cpu"), 0,
                                        val_entries=e[4:], lr=1e-3, weight_decay=0,
                                        dropout=0.3, batch_size=2)
        assert isinstance(model, PinClassifier) and vl >= 0 and 0 <= va <= 1

    def test_no_validation(self, tiny_dataset):
        d, e = tiny_dataset
        model, loss, acc = train_new_model(e, d, 2, torch.device("cpu"), 0,
                                           lr=1e-3, weight_decay=0, dropout=0.3, batch_size=4)
        assert loss >= 0 and acc == 0.0
