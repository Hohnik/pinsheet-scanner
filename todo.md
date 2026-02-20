# TODO

## Active

(none)

## Deferred

- [ ] Sheet 004 support — different pin diagram style (dashes), needs YOLO retraining
- [ ] OCR mismatches could trigger automatic re-labeling suggestions
- [ ] Integration tests (8 skipped, require model weights on disk)

## Done

- [x] **002 regression fixed** — retrained model predicts 472 ✓ (was 474).
      Model generalised past 2 noisy pseudo-labels without manual correction.
      OCR cross-validation confirms: C6R0 and C7R0 flags gone.
- [x] **100% accuracy** on all 593 crops (5337 pins) including manual-only (117 crops)
- [x] **All sheets match printed GESAMTERGEBNIS**: 001=335, 002=472, 003=454, 005=452, original=499
- [x] OCR accuracy: grayscale input, 3× upscale, confidence ≥60 filtering.
      0 false flags on 001/005/original, 3 residual OCR misreads across all sheets.
- [x] README rewritten to match actual architecture and CLI
- [x] Version bump 0.1.0 → 0.2.0, removed stale `flameprof` dependency
- [x] Labeler: smart sort (disagreements first), tags, transparent overlay
- [x] OCR parallel via ThreadPoolExecutor (~1.9s for 90 ROIs)
- [x] Flame graph via pyinstrument (replaced broken flameprof)
- [x] **15.4× scan speedup**: OCR opt-in, model LRU cache, batch CPU transfer
- [x] `collect --overwrite`, `accuracy --manual-only`, model backup, hyperparams provenance
- [x] YOLO min-detection guard, TTA inference fixes, train accuracy display
- [x] PinClassifier: 3-layer backbone + global context (~60K params)
- [x] Dataset expanded 120 → 593 via `collect` + pseudo-labels
- [x] Labeler UX: pin overlay on image, save-on-navigate, counter, speed timer
- [x] YOLO-first detection, per-Bahn summary
- [x] Training: cosine warmup, label smoothing, grad clip, AdamW
- [x] Codebase cleanup 3786 → 2392 lines
