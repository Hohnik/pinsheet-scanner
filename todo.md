# TODO

## Done

- [x] Write proper README.md
- [x] Consolidate tests (134 → 54)
- [x] Trim all docstrings and comments
- [x] Delete `src/__init__.py`, `src/constants.py`
- [x] Remove legacy `PinClassifier` (no backward compat)
- [x] New `PinClassifier`: shared backbone + spatial patch extraction
- [x] Simplify training: cosine annealing only, AdamW, no sidecar/experiment log
- [x] Merge `train_and_evaluate` + `retrain_all` → `train_new_model`
- [x] Remove `scikit-learn` dependency (inline KFold)
- [x] Remove 4-scheduler system
- [x] Clean up justfile
- [x] Progress bars for training epochs
- [x] Labeler UX redesign: pin overlay on image, save-on-navigate, counter, timer

## Active

- [ ] Label the 53 missing crops from sheet 002 to fix the 474→472 regression
      `just label` → arrow through, fix the ones that are wrong
- [ ] Retrain after labeling: `just train --epochs 200`
- [ ] Verify 002 == 472 again with `just scan sheets/002.jpeg`

## Code Quality / Correctness

- [ ] **`collect` skips already-labeled entries even when the model has improved**
      Add `--overwrite` flag so pseudo-labels can be refreshed after retraining.

- [ ] **YOLO minimum-detection safety check missing**
      `detect_pin_diagrams` returns YOLO results even if it finds only 1–2 diagrams
      (instead of the expected 90–120). Add a `min_yolo` threshold that triggers
      the classical fallback if too few boxes are returned.

- [ ] **`accuracy` conflates pseudo-labels with manual labels**
      100% accuracy is partly because pseudo-labels were produced by the same model.
      Add `--manual-only` flag that filters to crops whose filename prefix matches
      the manually labeled sheet (`original_*` by default).

- [ ] **Model backup before overwrite**
      `train` silently overwrites `models/pin_classifier.pt`.
      Rename old weights to `pin_classifier.YYYYMMDD_HHMMSS.pt.bak` before saving.

- [ ] **Training final-retrain shows `acc=0.0%`**
      No val loader in the last retrain → `acc` is always zero in the tqdm bar.
      Show train accuracy instead (compute it on the training batch after each epoch).

- [ ] **`collect` appends rows unsorted; `save_labels` sorts alphabetically**
      Running `collect` then `label` produces different CSV orderings.
      Either always sort on write (in `collect` too) or document the inconsistency.

- [ ] **`_TTA_CFG` inherits 10% scale jitter from `AugmentConfig` default**
      `scale_range=(0.9, 1.1)` is not explicitly set in `_TTA_CFG`, so the class
      default applies. Scale jitter at test time can shift pin positions; set
      `scale_range=(1.0, 1.0)` in the TTA config to disable it.

- [ ] **`hyperparams.json` has no provenance metadata**
      After re-tuning with a different architecture or dataset size, there is no
      record of which run produced the stored params. Append a small metadata block
      (date, sample count, model param count) to the JSON on each `tune` save.

## Deferred / Low Priority

- [ ] Sheet 004 support — different pin diagram style (dashes), needs YOLO retraining
- [ ] OCR mismatches unused — could trigger automatic flagging for manual review
- [ ] Extract labeler HTTP server into its own module (`src/label_server.py`)
- [ ] Integration tests (currently 8 skipped, require model weights on disk)
