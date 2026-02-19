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
