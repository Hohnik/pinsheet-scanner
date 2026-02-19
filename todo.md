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

## Low Priority

- [ ] Progress bars for training epochs
- [ ] Extract labeler HTTP server into its own module
