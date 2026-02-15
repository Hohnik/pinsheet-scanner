# Project: pinsheet-scanner

**Last Updated:** 2026-02-14

## Overview

Extract 9-pin bowling (Kegeln) scores from scanned score sheets using YOLO v11n for pin diagram detection and classical CV for pin state classification.

## Technology Stack

- **Language:** Python 3.13+
- **Package Manager:** uv
- **Detection:** ultralytics (YOLOv11n)
- **Image Processing:** OpenCV (opencv-python), NumPy
- **OCR:** pytesseract (requires system `tesseract` binary)
- **Testing:** pytest
- **Linting:** ruff
- **Type Checking:** basedpyright

## Directory Structure

```
src/pinsheet_scanner/
├── __init__.py         # CLI entry point (main), public API exports
├── classify.py         # Pin state classification (circular masks, intensity thresholds)
├── detect.py           # YOLO detection, bounding box sorting, cropping
├── ocr.py              # OCR-based score extraction for ground truth validation
└── pipeline.py         # Top-level process_sheet(), ThrowResult/SheetResult dataclasses

scripts/
├── debug_crops.py      # Debug visualization of cropped diagrams
└── train.py            # YOLO model training

data/                   # YOLO training/validation data (images + labels)
models/                 # Trained model weights (.pt, git-ignored)
debug_crops/            # Debug output: raw, cleaned, resized, overlay crops
tests/                  # pytest test suite
```

## Key Architecture

Pipeline: `detect → sort → crop → classify`

1. **detect.py** — `Detection` dataclass (x_center, y_center, width, height, confidence). Properties: `x_min`, `y_min`, `x_max`, `y_max`. `sort_detections()` assigns column/row indices.
2. **classify.py** — 9 circular masks at `PIN_POSITIONS` on 32x32 grid. `pins_from_diagram()` returns list of 9 binary values. `classify_pins_adaptive()` for threshold adaptation.
3. **pipeline.py** — `process_sheet()` orchestrates full pipeline. Returns `SheetResult` with list of `ThrowResult`.

## Development Commands

| Task | Command |
|------|---------|
| Install | `uv sync` |
| Test | `uv run pytest -q` |
| Lint | `ruff check . --fix && ruff format .` |
| Type check | `basedpyright src` |
| Train model | `uv run python -m scripts.train` |
| Scan sheet | `uv run pinsheet-scanner sheet.jpg` |
| Debug crops | `uv run python -m scripts.debug_crops` |

## System Dependencies

- **tesseract** — Required for OCR features (`brew install tesseract` on macOS)

## Notes

- Model weights (`.pt`) are git-ignored; must be trained locally
- Pin layout is diamond pattern: positions 0-8 mapped to `pins_down` list
- `debug_crops/` contains diagnostic output for classification tuning
