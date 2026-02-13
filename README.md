# Pinsheet Scanner

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using a small YOLO model for detection and classical CV for pin classification.

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
│  Crop & norm │  Extract each diagram, resize to 32×32
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Classify    │  9 circular masks → threshold intensity → pin state
└──────────────┘
```

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Hohnik/pinsheet-scanner.git
cd pinsheet-scanner
uv sync
```

## Usage

### 1. Label training data

Label pin diagrams in your scanned sheets using [Label Studio](https://labelstud.io/) or `label-studio start`
Export annotations in YOLO format and place them in `data/`:

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

### 2. Train the model

```bash
uv run python -m scripts.train
```

This produces a model under `runs/` which you copy to `models/pin_diagram.pt`.

### 3. Scan a sheet

```python
from pathlib import Path
from pinsheet_scanner import process_sheet

results = process_sheet(Path("sheet.jpg"))

for throw in results:
    print(
        f"Col {throw['column']}, Row {throw['row']}: "
        f"{throw['pins_down']} ({sum(throw['pins_down'])} pins)"
    )
```

### CLI

```bash
uv run pinsheet-scanner sheet.jpg
```

## Pin layout

`pins_down` returns a list of 9 binary values (1 = knocked down, 0 = standing)
mapped to the standard diamond layout:

```
    0
  1   2
3   4   5
  6   7
    8
```

## Project structure

```
pinsheet-scanner/
├── data/
│   └── dataset.yaml          # YOLO dataset config (template)
├── models/                    # Trained model weights (git-ignored)
├── scripts/
│   ├── label_helper.py        # Visualize / verify labels
│   └── train.py               # Train the YOLO model
├── src/
│   └── pinsheet_scanner/
│       ├── __init__.py
│       ├── detect.py          # YOLO detection + sorting
│       ├── classify.py        # Pin state classification (classical CV)
│       └── pipeline.py        # Top-level process_sheet()
├── tests/
│   ├── conftest.py
│   ├── test_detect.py
│   └── test_classify.py
├── pyproject.toml
└── README.md
```

## License

MIT
