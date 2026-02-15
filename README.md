# Pinsheet Scanner

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using a small YOLO model for detection and a tiny CNN for pin state classification.

## Architecture

```
Scan image → YOLO v11n → Sort boxes → Crop & binarise → CNN classifier → per-pin states
```

The YOLO detector finds pin diagram bounding boxes on the score sheet. Each crop is resized to 64×64 and binarised (Otsu threshold), then fed to a tiny CNN (`PinClassifier`, ~242k params) that independently predicts the state of each of the 9 pins.

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Hohnik/pinsheet-scanner.git
cd pinsheet-scanner
just install       # or: uv sync --all-extras
```

## Quick start

### 1. Train the YOLO detector

Label pin diagrams in your scanned sheets in YOLO format, place them in `data/`, then:

```bash
just train-detector
```

Copy the best weights to `models/pin_diagram.pt`.

### 2. Train the CNN classifier

```bash
just debug-crops sheet.jpg   # extract crops from a score sheet
just label                   # label them in the browser UI (1-9 toggle, Enter saves)
just train-classifier        # train on labeled crops with online augmentation
just accuracy                # measure per-pin and per-diagram accuracy
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

`pins_down` is a list of 9 binary values (1 = knocked down, 0 = standing) mapped to the standard diamond layout:

```
    0
  1   2
3   4   5
  6   7
    8
```

## Tasks

Run `just` to see all available tasks.

## License

MIT