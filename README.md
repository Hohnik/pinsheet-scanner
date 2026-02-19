# Pinsheet Scanner

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using computer vision and deep learning.

## Features

- **Automatic pin diagram detection** — classical blob analysis with YOLO fallback
- **CNN pin state classification** — spatial ROI classifier with test-time augmentation
- **OCR cross-validation** — optional Tesseract-based score verification
- **Sheet preprocessing** — perspective correction and contrast normalisation
- **Browser-based labeling UI** — annotate ground truth with confidence-sorted active learning
- **Hyperparameter tuning** — Optuna-powered search with experiment logging

## Pin Layout

The score sheet uses a 9-pin diamond pattern:

```
      8
    6   7
  3   4   5
    1   2
      0
```

Index 0 is the front (nearest) pin; numbering increases toward the back, left-to-right within each row.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                   # core dependencies
uv sync --extra dev       # + pytest, ruff
uv sync --extra ocr       # + pytesseract (optional)
```

For OCR support, also install [Tesseract](https://github.com/tesseract-ocr/tesseract):

```bash
brew install tesseract     # macOS
```

## Quick Start

```bash
# Scan a score sheet
just scan sheets/001.jpeg

# Extract crops for labeling
just extract sheets/001.jpeg

# Label ground truth (opens browser UI)
just label

# Tune hyperparameters
just tune

# Train the classifier (K-fold cross-validation + retrain)
just train
```

## Architecture

```
raw photo
  → preprocess.py    perspective correction + CLAHE
  → detect.py        classical blob detection (YOLO fallback)
  → classify.py      spatial ROI CNN + TTA
  → ocr.py           optional Tesseract cross-validation
  → pipeline.py      orchestrates the full flow
```

### Models

| Model | File | Purpose |
|-------|------|---------|
| `SpatialPinClassifier` | `models/pin_classifier.pt` | Per-pin state classification |
| YOLOv11n | `models/pin_diagram.pt` | Fallback diagram detection |

### CLI Commands

| Command | Description |
|---------|-------------|
| `scan` | Scan a score sheet and print per-throw results |
| `extract` | Extract pin diagram crops from a sheet image |
| `label` | Open browser labeling UI for ground truth annotation |
| `train` | K-fold cross-validate and retrain the CNN classifier |
| `tune` | Hyperparameter search with Optuna |
| `train-detector` | Train the YOLO pin diagram detector |
| `accuracy` | Validate CNN predictions against ground truth |

## Development

```bash
just test          # run unit tests
just integration   # run integration tests (requires model weights)
just lint          # lint and format with ruff
```
