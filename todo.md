# TODO

## Done

- [x] Write proper README.md
- [x] Reduce training.py duplication (shared `_run_training_loop`)
- [x] Fix module-level mutable RNG in classify.py
- [x] Remove PinClassifier legacy code → kept for backward compat with existing weights
- [x] Delete `src/__init__.py` (heavy imports, not useful for CLI tool)
- [x] Consolidate tests (134 → 61, removed redundant/obvious tests)
- [x] Trim all docstrings and comments
- [x] Remove over-abstracted `TrainingComponents`/`KFoldResult`/`kfold_train`
- [x] Remove `debug` param from `process_sheet`
- [x] Remove unnecessary loggers (debug-level only)
- [x] Rename `classify_pins_batch_with_confidence` → `classify_pins_batch`
- [x] Rename `test_tune.py` → `test_training.py`
- [x] Clean up justfile

## Low Priority

- [ ] Progress bars for training epochs
- [ ] Extract labeler HTTP server into its own module
