# TODO

## Active

- [ ] Label the 2 wrong pseudo-labels that cause the 002 regression:
      `just label` → find `002_c06_r00.png` and `002_c07_r00.png`, correct pin 8
      (model predicts down=1, label says up=0 — verify visually which is right)
- [ ] Retrain after correction: `just train --epochs 200`
- [ ] Verify 002 == 472 with `just scan sheets/002.jpeg`

## Deferred / Low Priority

- [ ] Sheet 004 support — different pin diagram style (dashes), needs YOLO retraining
- [ ] OCR mismatches (`--ocr` flag) currently informational — could trigger auto-flagging
- [ ] Integration tests (currently 8 skipped, require model weights on disk)

## Done

- [x] Write proper README.md
- [x] Consolidate tests (134 → 54)
- [x] Delete `src/__init__.py`, `src/constants.py`
- [x] New `PinClassifier`: shared backbone + spatial patch extraction (~60K params)
- [x] Simplify training: cosine annealing, AdamW, warmup, label smoothing, grad clip
- [x] Expand dataset 120 → 593 samples via `collect` + pseudo-labels
- [x] Labeler UX redesign: pin overlay on image, save-on-navigate, counter, timer
- [x] YOLO-first detection; classical as fallback
- [x] Per-Bahn summary in scan output
- [x] **Performance — 15.4× speedup** via profiling:
      - OCR opt-in (`scan --ocr`, default off) — was 91.5% of wall time
      - Batch OCR: single tesseract call per sheet (stacked ROI image)
      - Model LRU cache: YOLO + CNN cached across `process_sheet` calls (119× fewer `.to()`)
      - Batch `.cpu()` transfer: 90 per-item → 2 bulk (45× fewer round-trips)
- [x] `_TTA_CFG` cutout + gamma disabled at inference time (correctness fix)
- [x] Training final-retrain shows real train accuracy (was always 0.0%)
- [x] `collect --overwrite` — refresh pseudo-labels after retraining
- [x] YOLO min-detection guard (`min_yolo=10`) — prevent silent partial failures
- [x] `accuracy --manual-only` — honest ground-truth evaluation (no pseudo-label inflation)
- [x] Model backup before overwrite on `train` (timestamped `.bak.pt`)
- [x] `collect` uses `save_labels()` — consistent sorted CSV
- [x] `hyperparams.json` provenance (`_tuned_at`, `_samples`, `_model_params`)
