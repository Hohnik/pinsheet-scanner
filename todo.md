# TODO

## Active

- [ ] Label the 2 wrong pseudo-labels that cause the 002 regression:
      `just label` → `002_c06_r00.png` and `002_c07_r00.png` appear first
      (model predicts pin 8 = down, label says up — verify visually)
- [ ] Retrain after correction: `just train --epochs 200`
- [ ] Verify 002 == 472 with `just scan sheets/002.jpeg`

## Deferred

- [ ] Sheet 004 support — different pin diagram style (dashes), needs YOLO retraining
- [ ] OCR mismatches could trigger automatic re-labeling suggestions
- [ ] Integration tests (8 skipped, require model weights on disk)

## Done

- [x] README rewritten to match actual architecture and CLI
- [x] Version bump 0.1.0 → 0.2.0, removed stale `flameprof` dependency
- [x] Labeler: smart sort (disagreements first), tags, transparent overlay
- [x] OCR hybrid batch+parallel — 406 ms/sheet (14.5× faster than serial)
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
