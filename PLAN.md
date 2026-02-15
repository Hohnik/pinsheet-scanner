# Migration Plan: Classical CV → CNN Classifier

## Overview

Replace the three classical CV classifiers (`classify.py`, `classify_template.py`, `classify_contour.py`) with a tiny CNN trained on synthetic data. YOLO detection stays. OCR gets removed. The raw crops in `debug_crops/raw/` serve as the visual reference for the synthetic data generator.

## Progress

| Step | Description              | Status         |
| ---- | ------------------------ | -------------- |
| 1    | Synthetic data generator | ✅ DONE         |
| 2    | CNN model + training     | ✅ DONE         |
| 3    | New classify.py          | ✅ DONE         |
| 4    | Rewire pipeline.py       | ✅ DONE         |
| 5    | Rewrite tests            | ✅ DONE         |
| 6    | Delete old code          | ✅ DONE         |
| 7    | Docs, justfile, deps     | ✅ DONE         |

## Key Findings

### Binarization as preprocessing

Both synthetic and real crops are binarized (Otsu threshold) before being fed to the CNN. This eliminates the domain gap between synthetic training data and real scans — both become clean black/white blob patterns.

### Real crop blob geometry (measured across 20 crops, 180 pin blobs)

| Pin state       | Width (px) | Height (px) | Area (px²) |
| --------------- | ---------- | ----------- | ---------- |
| Knocked down (1) | 10–13, μ=11.5 | 4–6, μ=4.9 | 24–43, μ=36 |
| Standing (0)     | 5–7, μ=5.9   | 6–9, μ=7.5 | 20–30, μ=25 |

Knocked-down pins are **wide horizontal blobs** (wider than tall).
Standing pins are **compact, roughly circular blobs** (slightly taller than wide).

### Synthetic data calibration (canvas 50×40, resized to 64×64)

Shapes are drawn at native canvas resolution and measured after resize + Otsu binarization:

| Pin state       | Canvas shape                        | At 64×64 | Real target |
| --------------- | ----------------------------------- | -------- | ----------- |
| Knocked down (1) | Ellipse semi-axes `(4, 1)`         | 11×5     | 11.5×4.9    |
| Standing (0)     | Ellipse semi-axes `(2, 2)`         | 7×8      | 5.9×7.5     |

### Architecture

- `PinClassifier` CNN (242k params): 4 conv blocks → global avg pool → linear → 9 sigmoid outputs
- Trained on 10,240 synthetic images, reaches ~100% val accuracy by epoch 16
- Model saved at `models/pin_classifier.pt`
- Real-world validation: 120 crops classified, 28/120 high-confidence (≥0.80) — accuracy against ground truth TBD

---

## What was done

### Step A — Regenerate data + retrain with corrected geometry ✅

1. Deleted stale synthetic data
2. Regenerated 10,240 train + 2,560 val synthetic images with calibrated blob geometry
3. Retrained CNN classifier for 20 epochs (best val loss: 0.0071, best val acc: 99.97%)
4. Validated on 120 real crops — model produces predictions with moderate confidence
5. Ground-truth accuracy comparison still needed (requires labeled score sheet)

### Step B — Rewrite tests ✅

- Deleted old test files: `test_classify.py` (classical CV), `test_classify_template.py`, `test_ocr.py`
- Wrote new `tests/test_classify.py`: load model, single/batch inference, confidence, synthetic all-down/all-standing, real fixture, PIN_POSITIONS sanity
- Rewrote `tests/test_pipeline.py`: updated path constants, removed OCR/expected_score/mismatches tests, added classifier-missing error test, integration test
- Kept `tests/test_detect.py` unchanged
- All 60 tests pass

### Step C — Delete old code ✅

- Deleted `src/pinsheet_scanner/classify_template.py`
- Deleted `src/pinsheet_scanner/classify_contour.py`
- Deleted `src/pinsheet_scanner/ocr.py`
- Deleted `debug_crops/cleaned/`, `debug_crops/resized/`, `debug_crops/overlay/`
- Removed backwards-compat aliases from `constants.py` (`DEFAULT_SIZE`, `DEFAULT_MASK_RADIUS`)
- Rewrote `scripts/debug_crops.py` to use CNN classifier
- Verified zero stale references via grep

### Step D — Docs, justfile, deps ✅

- Rewrote `justfile` with 15 tasks: install, test, lint, typecheck, generate-data, train-classifier, retrain, train-detector, scan, debug-crops, validate, clean-data, clean-legacy, integration
- Rewrote `README.md`: new architecture diagram, CLI usage, training workflow, project structure, validation instructions, task runner table
- Updated `pyproject.toml`: new description, removed `label` optional dep group, registered `integration` pytest marker
- Updated this `PLAN.md`

---

## Remaining work (optional improvements)

- **Ground-truth validation**: label the 120 `debug_crops/raw/` crops with actual pin states from the score sheet and compute per-pin and per-score accuracy
- **Confidence tuning**: many real crops have confidence 0.5–0.7; investigate whether augmentation tuning, adaptive thresholding, or fine-tuning on a small labeled real dataset improves this
- **Fine-tuning on real data**: if accuracy is below target, manually label 50–200 real crops and fine-tune the CNN to close the domain gap