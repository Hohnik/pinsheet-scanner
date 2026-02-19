# Logbook

---

## 2026-02-19

### Pin numbering convention reversed (front = 0)
Flipped the pin index convention: index 0 is now the front (nearest) pin; numbers
increase toward the back, left-to-right within each row.
```
before:  0 / 1 2 / 3 4 5 / 6 7 / 8
after:   8 / 6 7 / 3 4 5 / 1 2 / 0
```
Remap applied: `new[i] = old[[8,6,7,3,4,5,1,2,0][i]]` (self-inverse permutation).
Migrated 120 rows in `debug_crops/labels.csv` (backup: `labels.csv.bak`).
Updated `constants.py`, `classify.py` docstring, `README.md`, and labeler `PIN_POS`.
CNN must be retrained with the new labels. All 134 unit tests pass.

### Full codebase analysis — improvement candidates documented

Analysed every source file and all 5 real score-sheet scans. Key findings captured
in `todo.md`. Highest-impact areas: spatial-ROI classifier, printed-score OCR
cross-validation, sheet pre-processing (perspective correction), synthetic data
generation, and detaching the labeler HTML template.

## 2026-02-19 (continued)

### Flat src/ package structure restored
`src/pinsheet_scanner/` subdirectory removed. All modules live directly in
`src/` with bare (non-relative) imports. Build backend switched from `uv_build`
to `hatchling` (`sources = {"src" = ""}`). Entry point changed to `cli:app`.
Added `pythonpath = ["src"]` to pytest config. All 134 tests pass.

### A1 — SpatialPinClassifier
New `SpatialPinClassifier` in `model.py`. Extracts a 12×12 patch at each of
the 9 known pin positions (constants: `PIN_COORDS_64`, `PATCH_SIZE`), encodes
with a shared CNN, and classifies each pin independently via a shared linear
head. `training.py` updated to train `SpatialPinClassifier` by default.

### A2 — Sheet pre-processing pipeline
New `preprocess.py`: `find_sheet_quad` + `rectify_sheet`. Detects the table
border via Canny+contours, applies a perspective warp to 1200×1600, then
CLAHE contrast normalisation. Gracefully falls back to CLAHE-only if no clean
quad is found. Wired into `pipeline.py` as the first step.

### A3 — Classical grid detector with YOLO fallback
`detect.py` gains `detect_pin_diagrams_classical`: Otsu binarise → keep
dot-sized blobs → morphological close → filter merged blobs by size/aspect.
Old `detect_pin_diagrams` → `detect_pin_diagrams_yolo`. Public
`detect_pin_diagrams` tries classical first (≥6 diagrams), falls back to YOLO.
YOLO detector now truly optional; no longer required to run `scan`.

### C4 — TTA always-on
`classify.py`: every inference call runs 5 forward passes (pass 1 clean,
passes 2–5 with mild augmentation) and averages sigmoid probabilities.
No flag — always active.

### C2 — Model bundle (sidecar JSON)
`training.py`: `save_model_bundle` saves `<weights>.json` alongside each `.pt`
checkpoint containing architecture name, git SHA, timestamp, and validation
accuracy. `load_classifier` reads the sidecar to decide which model class to
instantiate — fully backward compatible with weight-only `.pt` files.

### C3 — Experiment log
`training.py`: `log_experiment` appends a JSONL record to `experiments.jsonl`
after every `train` or `tune` run (timestamp, git SHA, hyperparams, metrics).

### D1 — OCR cross-validation
New `ocr.py`: `read_score_adjacent` runs pytesseract on the region to the left
of each detected diagram and parses the printed score digit. `cross_validate`
returns indices where `sum(cnn_pins) != ocr_score`. Wired into `pipeline.py`;
mismatches are exposed as `ThrowResult.ocr_mismatch`. Requires `[ocr]` extra.

### C1 — Labeler HTML extracted
HTML/JS template moved from `cli.py` string to `src/labeler.html`. Loaded at
runtime via `Path(__file__).parent / "labeler.html"`. CLI module reduced by
~300 lines.

### Q1 — Active-learning labeler ordering
`label` command sorts crops so unlabeled items come first, ordered by
ascending CNN confidence (least certain first). Already-labeled crops are
appended at the end. Crops list in `labeler.html` is pre-sorted server-side.

### CLI simplified — 8 commands → 7, many flags removed
Removed: `train-classifier`, `kfold`, `debug-crops`.
Added: `train` (merges kfold + retrain), `extract` (renamed debug-crops).
Removed flags: `--device`, `--seed`, `--no-open`, `--port`, `--no-retrain`,
`--val-count`. `justfile` updated accordingly.
