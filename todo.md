# TODO

## Active
- [ ] Retrain model on fresh crops (600 samples with correct aspect-ratio rectification)

## Deferred
- [ ] OCR mismatches could trigger automatic re-labeling suggestions
- [ ] Integration tests (8 skipped, require model weights on disk)

## Done

- [x] **Project restructure** — `debug_crops/` → `data/classifier/`, YOLO → `data/detector/`
- [x] **Cutout augmentation removed** — can't learn from hidden pins
- [x] **Multi-image `scan`/`collect`** — `just scan sheets/*`, `just collect sheets/*`
- [x] **Sheet 004 fixed** — aspect-ratio-preserving rectification. 30/30 detected, 108 total ✓
- [x] **002 regression fixed** — retrained model predicts 472 ✓ (was 474).
- [x] **100% accuracy** on all 600 crops (5400 pins)
- [x] **All sheets match printed GESAMTERGEBNIS**: 001=335, 002=472, 003=454, 004=108, 005=452, original=499
- [x] OCR accuracy: grayscale input, 3× upscale, confidence ≥60 filtering.
- [x] README rewritten to match actual architecture and CLI
- [x] Labeler: smart sort (disagreements first), tags, transparent overlay
- [x] **15.4× scan speedup**: OCR opt-in, model LRU cache, batch CPU transfer
- [x] `collect --overwrite`, `accuracy --manual-only`, model backup, hyperparams provenance
- [x] YOLO min-detection guard, TTA inference fixes, train accuracy display
- [x] PinClassifier: 3-layer backbone + global context (~60K params)
- [x] Dataset expanded via `collect` + pseudo-labels
- [x] Training: cosine warmup, label smoothing, grad clip, AdamW
