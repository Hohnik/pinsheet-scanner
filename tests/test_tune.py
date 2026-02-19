"""Tests for the parameterized PinClassifier, LR scheduler factory, defaults loading, and train_and_evaluate."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from constants import NUM_PINS
from model import PinClassifier
from training import (
    DEFAULTS,
    load_defaults,
    make_scheduler,
    train_and_evaluate,
)

# ---------------------------------------------------------------------------
# PinClassifier — dropout parameterization
# ---------------------------------------------------------------------------


class TestPinClassifierDropout:
    def test_default_dropout_is_0_3(self):
        model = PinClassifier()
        # The dropout layer is the first element in self.head
        dropout_layer = model.head[0]
        assert isinstance(dropout_layer, nn.Dropout)
        assert dropout_layer.p == pytest.approx(0.3)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.2, 0.3, 0.5, 0.9])
    def test_custom_dropout(self, dropout: float):
        model = PinClassifier(dropout=dropout)
        dropout_layer = model.head[0]
        assert isinstance(dropout_layer, nn.Dropout)
        assert dropout_layer.p == pytest.approx(dropout)

    def test_output_shape_unchanged(self):
        for drop in (0.0, 0.3, 0.5):
            model = PinClassifier(dropout=drop)
            x = torch.randn(2, 1, 64, 64)
            out = model(x)
            assert out.shape == (2, NUM_PINS)

    def test_parameter_count_independent_of_dropout(self):
        counts = set()
        for drop in (0.1, 0.3, 0.5):
            model = PinClassifier(dropout=drop)
            counts.add(sum(p.numel() for p in model.parameters()))
        assert len(counts) == 1, "Param count should not depend on dropout"

    def test_zero_dropout_passes_all_activations_in_eval(self):
        model = PinClassifier(dropout=0.0)
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        a = model(x)
        b = model(x)
        torch.testing.assert_close(a, b)

    def test_forward_runs_in_train_mode(self):
        model = PinClassifier(dropout=0.5)
        model.train()
        x = torch.randn(4, 1, 64, 64)
        out = model(x)
        assert out.shape == (4, NUM_PINS)


# ---------------------------------------------------------------------------
# make_scheduler — factory
# ---------------------------------------------------------------------------


def _make_optimizer(lr: float = 1e-3) -> torch.optim.Adam:
    """Create a minimal Adam optimizer for testing."""
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=lr)


class TestMakeScheduler:
    @pytest.mark.parametrize("name", ["plateau", "cosine", "onecycle", "step"])
    def test_returns_scheduler_for_valid_names(self, name: str):
        optimizer = _make_optimizer()
        scheduler = make_scheduler(name, optimizer, epochs=10, steps_per_epoch=5)
        assert scheduler is not None

    def test_unknown_name_raises(self):
        optimizer = _make_optimizer()
        with pytest.raises(ValueError, match="Unknown scheduler"):
            make_scheduler("nonexistent", optimizer, epochs=10, steps_per_epoch=5)

    def test_plateau_type(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        optimizer = _make_optimizer()
        sched = make_scheduler("plateau", optimizer, epochs=10, steps_per_epoch=5)
        assert isinstance(sched, ReduceLROnPlateau)

    def test_cosine_type(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = _make_optimizer()
        sched = make_scheduler("cosine", optimizer, epochs=10, steps_per_epoch=5)
        assert isinstance(sched, CosineAnnealingLR)

    def test_onecycle_type(self):
        from torch.optim.lr_scheduler import OneCycleLR

        optimizer = _make_optimizer()
        sched = make_scheduler("onecycle", optimizer, epochs=10, steps_per_epoch=5)
        assert isinstance(sched, OneCycleLR)

    def test_step_type(self):
        from torch.optim.lr_scheduler import StepLR

        optimizer = _make_optimizer()
        sched = make_scheduler("step", optimizer, epochs=10, steps_per_epoch=5)
        assert isinstance(sched, StepLR)


# ---------------------------------------------------------------------------
# Scheduler stepping — smoke tests
# ---------------------------------------------------------------------------


class TestSchedulerStepping:
    """Verify each scheduler can be stepped without errors."""

    def test_plateau_step(self):
        optimizer = _make_optimizer()
        sched = make_scheduler("plateau", optimizer, epochs=10, steps_per_epoch=5)
        for _ in range(3):
            sched.step(0.5)  # ReduceLROnPlateau needs a metric

    def test_cosine_step_lowers_lr(self):
        optimizer = _make_optimizer(lr=1e-2)
        sched = make_scheduler("cosine", optimizer, epochs=10, steps_per_epoch=5)
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            optimizer.step()
            sched.step()
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_onecycle_step_per_batch(self):
        optimizer = _make_optimizer(lr=1e-3)
        sched = make_scheduler("onecycle", optimizer, epochs=5, steps_per_epoch=4)
        # OneCycleLR expects exactly epochs * steps_per_epoch steps
        for _ in range(5 * 4):
            optimizer.step()
            sched.step()

    def test_step_reduces_lr_at_boundary(self):
        optimizer = _make_optimizer(lr=1e-2)
        sched = make_scheduler("step", optimizer, epochs=30, steps_per_epoch=5)
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(15):
            optimizer.step()
            sched.step()
        reduced_lr = optimizer.param_groups[0]["lr"]
        assert reduced_lr == pytest.approx(initial_lr * 0.5)


# ---------------------------------------------------------------------------
# load_defaults — picks up saved hyperparams
# ---------------------------------------------------------------------------


class TestLoadDefaults:
    def test_returns_builtin_defaults_when_no_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr(
            "training.HYPERPARAMS_PATH", tmp_path / "nonexistent.json"
        )
        defaults = load_defaults()
        assert defaults == DEFAULTS

    def test_loads_saved_hyperparams(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import json

        hp = {
            "lr": 3e-3,
            "weight_decay": 0.0,
            "dropout": 0.4,
            "batch_size": 8,
            "scheduler": "plateau",
        }
        hp_path = tmp_path / "hyperparams.json"
        hp_path.write_text(json.dumps(hp))
        monkeypatch.setattr("training.HYPERPARAMS_PATH", hp_path)

        defaults = load_defaults()
        assert defaults["lr"] == 3e-3
        assert defaults["dropout"] == 0.4
        assert defaults["batch_size"] == 8
        assert defaults["scheduler"] == "plateau"

    def test_saved_values_override_builtins(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import json

        hp = {
            "lr": 1e-2,
            "weight_decay": 1e-3,
            "dropout": 0.1,
            "batch_size": 32,
            "scheduler": "cosine",
        }
        hp_path = tmp_path / "hyperparams.json"
        hp_path.write_text(json.dumps(hp))
        monkeypatch.setattr("training.HYPERPARAMS_PATH", hp_path)

        defaults = load_defaults()
        # Every saved value should differ from the builtin defaults
        assert defaults["lr"] != DEFAULTS["lr"]
        assert defaults["weight_decay"] != DEFAULTS["weight_decay"]
        assert defaults["dropout"] != DEFAULTS["dropout"]
        assert defaults["batch_size"] != DEFAULTS["batch_size"]
        assert defaults["scheduler"] != DEFAULTS["scheduler"]


# ---------------------------------------------------------------------------
# train_and_evaluate — smoke test
# ---------------------------------------------------------------------------


class TestTrainAndEvaluate:
    """Minimal integration test using synthetic data to verify the function runs end-to-end."""

    @pytest.fixture()
    def tiny_dataset(self, tmp_path: Path):
        """Create 6 tiny grayscale PNGs with dummy labels."""
        import cv2
        import numpy as np

        crops_dir = tmp_path / "crops"
        crops_dir.mkdir()

        entries: list[tuple[str, list[int]]] = []
        for i in range(6):
            name = f"crop_{i:02d}.png"
            img = np.full((32, 32), 128 + i * 10, dtype=np.uint8)
            cv2.imwrite(str(crops_dir / name), img)
            pins = [1 if (i + p) % 2 == 0 else 0 for p in range(NUM_PINS)]
            entries.append((name, pins))

        return crops_dir, entries

    def test_returns_loss_and_accuracy(self, tiny_dataset):
        crops_dir, entries = tiny_dataset
        train = entries[:4]
        val = entries[4:]

        val_loss, val_acc = train_and_evaluate(
            train_entries=train,
            val_entries=val,
            crops_dir=crops_dir,
            epochs=2,
            device=torch.device("cpu"),
            seed=0,
            lr=1e-3,
            weight_decay=0.0,
            dropout=0.3,
            batch_size=2,
            scheduler_name="plateau",
        )
        assert isinstance(val_loss, float) and val_loss >= 0
        assert isinstance(val_acc, float) and 0.0 <= val_acc <= 1.0

    @pytest.mark.parametrize("sched", ["plateau", "cosine", "onecycle", "step"])
    def test_all_schedulers_run(self, tiny_dataset, sched: str):
        crops_dir, entries = tiny_dataset
        val_loss, val_acc = train_and_evaluate(
            train_entries=entries[:4],
            val_entries=entries[4:],
            crops_dir=crops_dir,
            epochs=2,
            device=torch.device("cpu"),
            seed=0,
            lr=1e-3,
            weight_decay=0.0,
            dropout=0.3,
            batch_size=4,
            scheduler_name=sched,
        )
        assert val_loss >= 0
        assert 0.0 <= val_acc <= 1.0
