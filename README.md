# Pinsheet Scanner

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using computer vision and deep learning.

## Features

- **Automatic pin diagram detection** — classical blob analysis with YOLO fallback
- **CNN pin state classification** — shared-backbone spatial classifier with TTA
- **OCR cross-validation** — optional Tesseract-based score verification
- **Sheet preprocessing** — perspective correction and contrast normalisation
- **Browser-based labeling UI** — confidence-sorted active learning
- **Hyperparameter tuning** — Optuna-powered search

## Pin Layout

```
      8
    6   7
  3   4   5
    1   2
      0
```

Index 0 is the front pin; numbering increases toward the back.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                   # core dependencies
uv sync --extra dev       # + pytest, ruff
uv sync --extra ocr       # + pytesseract (optional)
```

## Quick Start

```bash
just scan sheets/001.jpeg       # scan a score sheet
just extract sheets/001.jpeg    # extract crops for labeling
just label                      # label ground truth (browser UI)
just tune                       # hyperparameter search
just train                      # k-fold cross-validate + retrain
```

## Architecture

```
raw photo
  → preprocess.py    perspective correction + CLAHE
  → detect.py        classical blob detection (YOLO fallback)
  → classify.py      PinClassifier CNN + TTA
  → ocr.py           optional Tesseract cross-validation
  → pipeline.py      orchestrates the full flow
```

### PinClassifier

Shared-backbone CNN with spatial pin extraction. Two conv layers produce
a 32-channel feature map at full resolution (64×64). A 12×12 patch is
extracted from the feature map at each of the 9 known pin positions,
average-pooled to a 32-dim vector, and classified by a shared linear head.

### CLI Commands

| Command | Description |
|---------|-------------|
| `scan` | Scan a score sheet and print results |
| `extract` | Extract pin diagram crops |
| `label` | Browser labeling UI |
| `train` | K-fold cross-validate + retrain CNN |
| `tune` | Optuna hyperparameter search |
| `train-detector` | Train YOLO detector |
| `accuracy` | Validate predictions against labels |

## Development

```bash
just test       # unit tests
just lint       # ruff check + format
```
