# Kegel Detector

Extract 9-pin bowling scores from scanned score sheets using computer vision.  
No ML required—classical OpenCV heuristics only.

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv):

```bash
uv pip install git+https://github.com/yourusername/kegel-detector.git
```

Or clone and sync locally:

```bash
git clone https://github.com/yourusername/kegel-detector.git
cd kegel-detector
uv sync
```

## Usage

```python
from pathlib import Path
from kegel_detector import process_sheet

results = process_sheet(Path("sheet.jpg"))

for throw in results:
    print(f"Bahn {throw['bahn']}, Throw {throw['throw']}: "
          f"{sum(throw['pins_down'])} pins down")
```

`pins_down` returns a list of 9 binary values (1 = knocked down, 0 = standing)  
mapped to the standard diamond layout:

```
    0
  1   2
3   4   5
  6   7
    8
```

## How it works

1. **Deskew** – Corrects scan rotation via Hough line detection
2. **Grid parse** – Extracts cells from the table structure (Bahn 1–4)
3. **Diagram isolation** – Crops the 9-pin dot pattern from each cell
4. **Heuristic detection** – Applies 9 circular masks; thresholding determines pin state
5. **Validation** – Cross-checks detected count against the printed numeric score
