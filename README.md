# Pinsheet Scanner

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using computer vision and deep learning.

## Features

- **YOLO pin diagram detection** with classical blob-analysis fallback
- **CNN pin state classification** — 3-layer backbone + spatial extraction + TTA (~60K params)
- **OCR cross-validation** — optional parallel Tesseract verification (`--ocr`)
- **Sheet preprocessing** — perspective correction and CLAHE contrast normalisation
- **Browser-based labeling UI** — disagreements-first sort, pin overlay on image
- **Hyperparameter tuning** — Optuna search with cosine-warmup schedule
- **~400 ms/sheet** without OCR, ~800 ms with OCR

## Pin Layout

```
      8           (back)
    6   7
  3   4   5
    1   2
      0           (front)
```

Index 0 is the front pin; numbering increases toward the back, left-to-right.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                   # core dependencies
uv sync --extra dev       # + pytest, ruff
uv sync --extra ocr       # + pytesseract (optional)
```

## Quick Start

```bash
just scan sheets/001.jpeg         # scan a score sheet
just scan sheets/001.jpeg --ocr   # with OCR cross-validation
just extract sheets/001.jpeg      # extract crops for labeling
just label                        # label ground truth (browser UI)
just tune                         # hyperparameter search (20 trials)
just train                        # k-fold cross-validate + retrain
just accuracy                     # full dataset accuracy
just accuracy --manual-only       # hand-labeled crops only
```

## Architecture

```
raw photo
  → preprocess.py    perspective correction + CLAHE
  → detect.py        YOLO detection (classical fallback if < 10 boxes)
  → classify.py      PinClassifier CNN + 5-pass TTA
  → ocr.py           optional parallel Tesseract cross-validation
  → pipeline.py      orchestrates the full flow (model LRU cache)
```

### PinClassifier

Shared-backbone CNN with spatial pin extraction and global context.
Three conv layers (1→32→64→64, all 3×3, padding-preserving) build a
64-channel feature map at full 64×64 resolution. For each of the 9 pin
positions, a 16×16 patch is extracted from the feature map and average-pooled
to a 64-dim local vector. A global average pool gives a 64-dim context vector.
Local + global are concatenated (128-dim) and fed to a shared 2-layer head
(128→32→1). Total: ~60K parameters.

### Dataset

593 labeled crops in `debug_crops/`: 120 hand-labeled from `original.jpeg`,
473 pseudo-labeled from sheets 001–003, 005 via `collect`.

### CLI Commands

| Command | Description |
|---------|-------------|
| `scan [--ocr]` | Scan a sheet, print per-throw results and Bahn summaries |
| `collect [--overwrite]` | Harvest high-confidence crops into the training set |
| `extract` | Extract pin diagram crops for inspection |
| `label` | Browser labeling UI (disagreements first) |
| `train` | K-fold cross-validate + retrain CNN (auto-backup old weights) |
| `tune` | Optuna hyperparameter search (saves provenance metadata) |
| `train-detector` | Train YOLO detector |
| `accuracy [--manual-only]` | Validate predictions against labels |

### Performance

Profiled with cProfile + pyinstrument (see `scripts/profile_scan.py`):

| Metric | Value |
|--------|-------|
| Scan (no OCR) | ~400 ms/sheet |
| Scan (with OCR) | ~800 ms/sheet |
| Training (200 epochs, 593 samples) | ~6 min on MPS |
| Tuning (20 trials × 20 epochs) | ~10 min on MPS |

## Development

```bash
just test           # 54 unit tests
just lint           # ruff check + format
just integration    # end-to-end tests (requires model weights)
```
