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

### Codebase quality improvements — refactoring round

Analysed all source modules (excluding tests) for quality issues. Implemented:

1. **README.md** — Replaced placeholder with full project docs (features, install,
   quick start, architecture overview, CLI command reference, dev instructions).

2. **training.py** — Extracted `TrainingComponents` dataclass and `build_training_components()`
   factory to eliminate duplicated model/optimizer/scheduler setup between
   `train_and_evaluate` and `retrain_all`. Added `KFoldResult` dataclass and
   `kfold_train()` function to move K-fold cross-validation logic out of the CLI.

3. **classify.py** — Removed module-level `_TTA_RNG` (global mutable state with fixed
   seed). TTA now creates a fresh `np.random.default_rng(42)` per batch call —
   thread-safe and deterministic.

4. **cli.py** — `train` command reduced from ~80 lines of K-fold logic to ~30 lines
   that delegate to `training.kfold_train()`. Added `_load_detection_pipeline()` shared
   helper for detection+crop setup. Added `_setup_logging()` for consistent log config.

5. **detect.py / pipeline.py** — Replaced bare `Any` type for YOLO model with `YOLOModel`
   type alias. Added `logging` calls for detection fallback decisions.

6. **preprocess.py** — Added `logging` for fallback when no sheet quad is found.

7. **`src/__init__.py`** — New file exposing clean public API (`process_sheet`,
   `load_classifier`, `classify_pins_batch_with_confidence`, `Detection`, etc.).

All 134 tests pass.

### Full cleanup — 3786 → 1852 lines (51% reduction)

Aggressive cleanup pass across all source and test files:

**Source (2497 → 1424 lines):**
- Deleted `src/__init__.py` — heavy import-time side effects, nobody uses it as a library
- `training.py` — Removed over-abstracted `TrainingComponents`, `build_training_components`,
  `KFoldResult`, `kfold_train`. Replaced `train_one_epoch` + `evaluate` with a single
  `_run_training_loop` used by both `train_and_evaluate` and `retrain_all`
- `classify.py` — Renamed `classify_pins_batch_with_confidence` → `classify_pins_batch`.
  Inlined `_confidence_from_probs`. Removed `AnyClassifier` type alias
- `pipeline.py` — Removed `debug` param (opened cv2.imshow, never used in practice).
  Removed logger. Removed section label comments (A2, A3, etc.)
- `model.py` — Inlined `_conv_block` into `PinClassifier.__init__`. Trimmed docstrings
- `detect.py` — Removed logger, trimmed docstrings and section separators
- `preprocess.py` — Removed logger, trimmed docstrings
- `augment.py`, `labels.py`, `ocr.py` — Trimmed docstrings throughout
- `cli.py` — Inlined K-fold loop (concise). Removed `_format_kfold_table`,
  `_setup_logging`. Simplified `accuracy` output. Renamed helpers
- `constants.py` — Removed unused `CLASSIFIER_INPUT_SIZE`
- `justfile` — Removed group tags, typecheck recipe

**Tests (1289 → 428 lines, 134 → 61 tests):**
- `test_labels.py` — 233→67: kept roundtrip + edge cases, removed 15+ redundant tests
- `test_preprocess.py` — 176→53: kept contract + determinism, removed pixel edge cases
- `test_augment.py` — 118→47: kept core behaviors, removed frozen/default tests
- `test_tune.py` → `test_training.py` — 290→80: removed PinClassifier dropout tests,
  scheduler type-checking tests. Kept smoke tests
- `test_classify.py` — 225→82: consolidated weight-dependent tests
- `test_detect.py` — 113→49: merged test classes
- `test_pipeline.py` — 134→50: removed TestDefaultPaths, trimmed

All 61 tests pass.

### Architecture & training overhaul — 1852 → 1679 lines

**Model (`model.py` 71→57 lines):**
- Deleted legacy `PinClassifier` (global-avg-pool, 4 conv blocks, 18K params)
- Renamed `SpatialPinClassifier` → `PinClassifier`
- New architecture: shared 2-layer backbone (32 channels, no pooling) produces
  a feature map at full 64×64 resolution. 12×12 patches extracted at each of
  the 9 pin positions from the *feature map* (not raw pixels), average-pooled
  to 32-dim vectors, classified by shared linear head. ~10K params.
- Absorbed `constants.py` — `NUM_PINS`, `PATCH_SIZE`, `PIN_COORDS` now in model.py

**Training (`training.py` 257→162 lines):**
- Removed 4-scheduler system (plateau/cosine/onecycle/step) → cosine annealing only
- Removed `make_scheduler`, `_step_scheduler`, `save_model_bundle`, `log_experiment`
- Removed sidecar JSON and `experiments.jsonl`
- AdamW replaces Adam (correct weight decay decoupling)
- Best weights kept in memory, restored at end (no disk I/O during training)
- Merged `train_and_evaluate` + `retrain_all` → single `train_new_model`
- Renamed `RealCropDataset` → `CropDataset`
- `split_entries` takes `val_fraction` instead of `val_count`

**Classify (`classify.py` 117→102 lines):**
- Removed legacy `PinClassifier` import and sidecar JSON detection
- `load_classifier` gives clear error on architecture mismatch
- Batch confidence computed in one line via `(avg - 0.5).abs().mean(dim=1) * 2`

**CLI (`cli.py` 407→398 lines):**
- K-fold inlined with numpy (removed `scikit-learn` dependency)
- Removed experiment logging
- `_hp_keys()` filters hyperparams dict for training-relevant keys only

**Dependencies:**
- Dropped `scikit-learn` — KFold implemented with `np.array_split`

**Labels (`labels.py`):**
- Removed `constants` dependency, hardcoded 9 (CSV format constant)

Retrained model: 100% accuracy across all 5 folds (120 images).
All 54 tests pass.
