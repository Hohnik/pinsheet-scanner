# Pinsheet Scanner

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using a small YOLO model for detection and a tiny CNN for pin state classification.

## Architecture

```
Scan image
    │
    ▼
┌──────────────┐
│  YOLO v11n   │  Detect pin diagram bounding boxes
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Sort boxes  │  Order by column (x) then row (y) → reading order
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Crop & bin  │  Extract each diagram, resize to 64×64, Otsu binarise
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  CNN class.  │  PinClassifier → 9 sigmoid outputs → per-pin state
└──────────────┘
```

The YOLO detector finds pin diagram bounding boxes on the score sheet. Each crop is resized and binarised (Otsu threshold), then fed to a tiny CNN (`PinClassifier`, ~242k params) that independently predicts the state of each of the 9 pins.

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Hohnik/pinsheet-scanner.git
cd pinsheet-scanner
just install       # or: uv sync --all-extras
```

## Quick start

### 1. Train the YOLO detector

Label pin diagrams in your scanned sheets in YOLO format and place them in `data/`:

```
data/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Only one class is needed: `pin_diagram`.

```bash
just train-detector              # or: uv run python -m scripts.train
```

Copy the best weights to `models/pin_diagram.pt`.

### 2. Train the CNN classifier

The classifier is trained entirely on synthetic data — no manual labelling required:

```bash
just retrain                     # clean → generate → train (all-in-one)

# Or step by step:
just clean-data                  # rm -rf data/classifier
just generate-data               # 10k+ synthetic crops
just train-classifier             # train CNN for 20 epochs
```

Weights are saved to `models/pin_classifier.pt`.

### 3. Scan a sheet

```bash
just scan sheet.jpg

# With extra flags:
just scan sheet.jpg --confidence 0.3 --classifier-model models/custom.pt
```

### Python API

```python
from pathlib import Path
from pinsheet_scanner import process_sheet

result = process_sheet(Path("sheet.jpg"))

for throw in result.throws:
    pins = "".join(str(p) for p in throw.pins_down)
    print(f"Col {throw.column}, Row {throw.row}: {pins} ({throw.score} pins)")

print(f"Total: {result.total_pins}")
```

## Pin layout

`pins_down` returns a list of 9 binary values (1 = knocked down, 0 = standing) mapped to the standard diamond layout:

```
    0
  1   2
3   4   5
  6   7
    8
```

## Validation

After training, validate the classifier on real crops extracted from a scan:

```bash
just debug-crops sheet.jpg       # extract raw crops → debug_crops/raw/
just validate                    # run CNN on all crops, print per-crop predictions
```

## Task runner

Run `just` to see all available tasks:

| Task | Description |
|---|---|
| `install` | Install all dependencies |
| `test` | Run unit tests |
| `integration` | Run integration tests (requires model weights) |
| `lint` | Lint and format with ruff |
| `typecheck` | Type-check with basedpyright |
| `generate-data` | Generate synthetic training data |
| `train-classifier` | Train the CNN pin classifier |
| `retrain` | Clean + generate + train (all-in-one) |
| `train-detector` | Train the YOLO detector |
| `scan <image>` | Scan a score sheet |
| `debug-crops <image>` | Save debug crops from a scan |
| `validate` | Validate classifier on `debug_crops/raw/` |
| `clean-data` | Delete synthetic data |
| `clean-legacy` | Remove obsolete pre-migration files |

## Project structure

```
pinsheet-scanner/
├── data/
│   ├── dataset.yaml              # YOLO dataset config
│   └── classifier/               # Synthetic training data (generated)
│       ├── train/
│       ├── val/
│       ├── train_labels.csv
│       └── val_labels.csv
├── debug_crops/
│   └── raw/                      # Real crops for validation
├── models/                       # Trained weights (git-ignored)
│   ├── pin_diagram.pt            # YOLO detector
│   └── pin_classifier.pt         # CNN classifier
├── scripts/
│   ├── debug_crops.py            # Extract + classify crops for debugging
│   ├── generate_data.py          # Synthetic data generator
│   ├── train.py                  # Train the YOLO detector
│   └── train_classifier.py       # Train the CNN classifier
├── src/
│   └── pinsheet_scanner/
│       ├── __init__.py            # CLI entry point + public API
│       ├── classify.py            # CNN inference (Otsu + forward pass)
│       ├── constants.py           # Pin positions, input size
│       ├── detect.py              # YOLO detection + sorting + cropping
│       ├── model.py               # PinClassifier CNN definition
│       └── pipeline.py            # Top-level process_sheet()
├── tests/
│   ├── fixtures/
│   │   └── sample_crop.png
│   ├── test_classify.py
│   ├── test_detect.py
│   └── test_pipeline.py
├── justfile
├── pyproject.toml
└── README.md
```

## License

MIT