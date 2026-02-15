# Project: pinsheet-scanner

## Overview

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using YOLO v11n for pin diagram detection and a tiny CNN for pin state classification.

## Technology Stack

- **Language:** Python 3.13+
- **Package Manager:** uv
- **Detection:** ultralytics (YOLOv11n)
- **Classification:** PyTorch (tiny CNN — `PinClassifier`, ~242k params)
- **Image Processing:** OpenCV (opencv-python), NumPy
- **Testing:** pytest
- **Linting:** ruff
- **Type Checking:** basedpyright
- **Task Runner:** just

## Architecture

Pipeline: `detect → sort → crop → preprocess → classify`

1. **detect.py** — YOLO inference returns `Detection` dataclasses. `sort_detections()` clusters by x-position into columns, sorts top→bottom within each column, and assigns column/row indices.
2. **classify.py** — `preprocess_crop()` resizes to 64×64, applies Otsu binarisation, normalises to [0,1]. `classify_pins_batch_with_confidence()` runs the CNN and returns per-pin binary states + confidence.
3. **model.py** — `PinClassifier`: 4-block CNN (Conv→BN→ReLU→Pool) + adaptive avg pool + linear head. Input: 1×64×64. Output: 9 logits (sigmoid → independent per-pin probabilities).
4. **pipeline.py** — `process_sheet()` orchestrates the full pipeline. Returns `SheetResult` with a list of `ThrowResult`.
5. **augment.py** — Online augmentation for training: brightness, noise, blur, rotation, scale jitter, grid-line remnants. Used by `train_classifier.py`.
6. **constants.py** — `PIN_POSITIONS` (diamond layout), `NUM_PINS` (9), `CLASSIFIER_INPUT_SIZE` (64×64).

## Directory Structure

```
src/pinsheet_scanner/
├── __init__.py         # CLI entry point (main), public API exports
├── augment.py          # Online augmentation pipeline for training
├── classify.py         # CNN inference + preprocessing
├── constants.py        # Pin layout, input size, shared constants
├── detect.py           # YOLO detection, spatial sorting, cropping
├── model.py            # PinClassifier CNN definition
└── pipeline.py         # Top-level process_sheet()

scripts/
├── debug_crops.py      # Extract + classify crops for debugging
├── label.py            # Browser-based labeling UI for ground truth
├── train.py            # YOLO detector training
├── train_classifier.py # CNN training on real labeled crops
└── validate_labels.py  # Accuracy measurement vs ground truth

data/                   # YOLO training/validation data (images + labels)
models/                 # Trained model weights (.pt, git-ignored)
debug_crops/
├── raw/                # Real crops (training data for classifier)
└── labels.csv          # Ground-truth pin labels
tests/                  # pytest test suite
```

## Development Commands

| Task | Command |
|------|---------|
| Install | `just install` or `uv sync --all-extras` |
| Test | `just test` or `uv run pytest -q` |
| Integration tests | `just integration` |
| Lint + format | `just lint` |
| Type check | `just typecheck` |
| Train classifier | `just retrain` |
| Train detector | `just train-detector` |
| Scan sheet | `just scan sheet.jpg` |
| Debug crops | `just debug-crops sheet.jpg` |
| Label crops | `just label` |
| Measure accuracy | `just accuracy` |

## Key Design Decisions

- **Real-crop training only** — synthetic rendering was tried and abandoned due to domain gap. The classifier trains on ~120 real labeled crops with heavy online augmentation.
- **Otsu binarisation** — preprocessing binarises crops to remove intensity variation across scans. Trade-off: may lose subtle information at diamond extremes (pins 0, 8).
- **Independent pin classification** — each pin is predicted independently (9 sigmoids, not a single multi-class output) because any combination of the 9 pins can be knocked down.
- **Lazy YOLO import** — `ultralytics` is imported inside `load_model()` to keep import time fast when only using the classifier.

## Notes

- Model weights (`.pt`) are git-ignored; must be trained locally.
- Pin layout is a diamond: indices 0–8 map to the `pins_down` list.
- `debug_crops/` is both diagnostic output and classifier training data.
- Current accuracy: 96.9% per-pin, 75.0% per-diagram on 120 labeled crops.