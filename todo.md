# TODO

## High Priority

- [x] **README.md** — Write a real project README (description, install, usage, architecture)
- [x] **Reduce duplication in training.py** — `train_and_evaluate` and `retrain_all` share 90% of their setup code; extract a shared `build_training_components` helper
- [x] **Fix module-level mutable RNG in classify.py** — `_TTA_RNG` is global mutable state with a fixed seed; create a fresh RNG per batch call instead
- [x] **Refactor cli.py train command** — 80+ lines of business logic (K-fold loop, printing, logging) moved to `training.py::kfold_train`; CLI is a thin shell
- [x] **Extract shared CLI helpers** — `_load_detection_pipeline` consolidates detection setup across `extract` and future commands
- [x] **Reduce cli.py extract duplication** — `extract` now uses `_load_detection_pipeline` shared helper

## Medium Priority

- [x] **Add structured logging** — Added `logging` module to `detect.py`, `pipeline.py`, `preprocess.py`, `cli.py`, `training.py`; CLI sets up logging via `_setup_logging()`
- [x] **Clean up type annotations** — Replaced `Any` for YOLO model with `YOLOModel` type alias in `detect.py` and `pipeline.py`
- [x] **Add `__init__.py`** — Expose clean public API for programmatic use

## Low Priority

- [ ] **Progress bars** — Add `tqdm` or similar for training epochs and Optuna trials
- [ ] **Labeler server cleanup** — The inline HTTP server in the `label` command could be a small standalone module
