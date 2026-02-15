# Project: pinsheet-scanner

## Overview

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using YOLO v11n for pin diagram detection and a tiny CNN for pin state classification.

## Technology Stack

- **Language:** Python 3.13+
- **Package Manager:** uv
- **CLI:** typer (unified `pinsheet-scanner` command with subcommands)
- **Detection:** ultralytics (YOLOv11n)
- **Classification:** PyTorch (tiny CNN — `PinClassifier`, ~242k params)
- **Image Processing:** OpenCV (opencv-python), NumPy
- **Tuning:** Optuna (TPE sampler)
- **Cross-validation:** scikit-learn (KFold)
- **Testing:** pytest
- **Linting:** ruff
- **Type Checking:** basedpyright
- **Task Runner:** just (run `just` to see all tasks)

## Architecture

Pipeline: `detect → sort → crop → preprocess → classify`

All source code lives in `src/pinsheet_scanner/`:

1. **detect.py** — YOLO inference returns `Detection` dataclasses. `sort_detections()` clusters by x-position into columns, sorts top→bottom within each column, and assigns column/row indices.
2. **classify.py** — `preprocess_crop()` resizes to 64×64, applies Otsu binarisation, normalises to [0,1]. `classify_pins_batch_with_confidence()` runs the CNN and returns per-pin binary states + confidence.
3. **model.py** — `PinClassifier`: 4-block CNN (Conv→BN→ReLU→Pool) + adaptive avg pool + linear head. Input: 1×64×64. Output: 9 logits (sigmoid → independent per-pin probabilities).
4. **pipeline.py** — `process_sheet()` orchestrates the full pipeline. Returns `SheetResult` with a list of `ThrowResult`.
5. **augment.py** — Online augmentation for training: brightness, noise, blur, rotation, scale jitter, grid-line remnants.
6. **labels.py** — Shared CSV I/O for ground-truth pin-state labels (used by training, validation, and labeling commands).
7. **constants.py** — `NUM_PINS` (9), `CLASSIFIER_INPUT_SIZE` (64×64).
8. **training.py** — Shared training primitives: dataset (`RealCropDataset`), scheduler factory, training loop (`train_and_evaluate`), hyperparameter persistence, train/val splitting.
9. **cli.py** — Unified typer CLI with all commands: `scan`, `train-classifier`, `train-detector`, `tune`, `kfold`, `debug-crops`, `label`, `accuracy`.

## Key Design Decisions

- **Unified CLI** — All commands live in `cli.py` as typer commands, exposed via the `pinsheet-scanner` console entry-point. No separate `scripts/` directory.
- **Real-crop training only** — synthetic rendering was tried and abandoned due to domain gap. The classifier trains on ~120 real labeled crops with heavy online augmentation.
- **Otsu binarisation** — preprocessing binarises crops to remove intensity variation across scans. Trade-off: may lose subtle information at diamond extremes (pins 0, 8).
- **Independent pin classification** — each pin is predicted independently (9 sigmoids, not a single multi-class output) because any combination of the 9 pins can be knocked down.
- **Lazy YOLO import** — `ultralytics` is imported inside `load_model()` to keep import time fast when only using the classifier.
- **Hyperparameter persistence** — `models/hyperparams.json` stores tuned defaults; `train-classifier` and `kfold` commands load these automatically (CLI flags still override).

## Notes

- Model weights (`.pt`) are git-ignored; must be trained locally.
- Pin layout is a diamond: indices 0–8 map to the `pins_down` list.
- `debug_crops/` is both diagnostic output and classifier training data.
- All commands are accessible via `just` tasks or directly as `pinsheet-scanner <command>`.