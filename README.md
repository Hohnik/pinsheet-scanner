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

**Current accuracy** (120 labeled crops, 100 train / 20 val):

| Metric               | Value              |
| -------------------- | ------------------ |
| Per-pin accuracy     | 96.9 % (1047/1080) |
| Per-diagram accuracy | 75.0 % (90/120)    |

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
just train-detector
```

Copy the best weights to `models/pin_diagram.pt`.

### 2. Train the CNN classifier

The classifier is trained on real labeled crops with heavy online augmentation. You need labeled data in `debug_crops/`:

```bash
# Extract crops from a score sheet
just debug-crops sheet.jpg

# Label the crops (opens browser UI)
just label

# Train the classifier (100 train / 20 val split)
just retrain
```

Weights are saved to `models/pin_classifier.pt`.

### 3. Scan a sheet

```bash
just scan sheet.jpg
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

## Labeling & validation

### Label ground truth

A built-in labeling UI lets you annotate the correct pin states for each crop. CNN predictions are pre-populated as suggestions — you just correct the mistakes.

```bash
just label
```

- Click pins or press `1`–`9` to toggle knocked-down/standing
- `Enter` saves and advances, `←`/`→` navigate
- Progress is auto-saved to `debug_crops/labels.csv`

### Measure accuracy

```bash
just accuracy
```

Reports per-pin accuracy, per-diagram accuracy, per-position breakdown, and a detailed mismatch list.

## Task runner

Run `just` to see all available tasks:

| Task                 | Description                                        |
| -------------------- | -------------------------------------------------- |
| `install`            | Install all dependencies                           |
| `test`               | Run unit tests                                     |
| `integration`        | Run integration tests (requires model weights)     |
| `lint`               | Lint and format with ruff                          |
| `typecheck`          | Type-check with basedpyright                       |
| `train-classifier`   | Train the CNN on real labeled crops                |
| `retrain`            | Retrain from scratch                               |
| `train-detector`     | Train the YOLO detector                            |
| `scan <image>`       | Scan a score sheet                                 |
| `debug-crops <image>`| Save debug crops from a scan                       |
| `label`              | Open labeling UI to annotate ground-truth          |
| `accuracy`           | Compare ground-truth labels against CNN predictions|
| `validate`           | Run CNN on real crops, print per-crop predictions  |

## Project structure

```
pinsheet-scanner/
├── data/
│   ├── dataset.yaml              # YOLO dataset config
│   ├── train/                    # YOLO training images + labels
│   └── val/                      # YOLO validation images + labels
├── debug_crops/
│   ├── raw/                      # Real crops (training data for classifier)
│   └── labels.csv                # Ground-truth labels
├── models/
│   ├── pin_diagram.pt            # YOLO detector weights
│   └── pin_classifier.pt         # CNN classifier weights
├── scripts/
│   ├── debug_crops.py            # Extract + classify crops for debugging
│   ├── label.py                  # Labeling server for ground-truth annotation
│   ├── train.py                  # Train the YOLO detector
│   ├── train_classifier.py       # Train the CNN classifier on real crops
│   └── validate_labels.py        # Compare labels against CNN predictions
├── src/
│   └── pinsheet_scanner/
│       ├── __init__.py            # CLI entry point + public API
│       ├── augment.py             # Online augmentation for training
│       ├── classify.py            # CNN inference
│       ├── constants.py           # Pin positions, input size
│       ├── detect.py              # YOLO detection + sorting + cropping
│       ├── model.py               # PinClassifier CNN definition
│       └── pipeline.py            # Top-level process_sheet()
├── tests/
│   ├── fixtures/
│   │   ├── all_down.png
│   │   ├── all_standing.png
│   │   └── sample_crop.png
│   ├── test_augment.py
│   ├── test_classify.py
│   ├── test_detect.py
│   └── test_pipeline.py
├── justfile
├── pyproject.toml
└── README.md
```

## License

MIT