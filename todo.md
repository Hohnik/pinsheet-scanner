# TODO

Tasks are grouped by theme and ordered by expected impact within each group.
Status: `[ ]` open · `[~]` in progress · `[x]` done

---

## Architecture

### A1 · Spatial ROI classifier [ ]
**Problem:** `PinClassifier` uses `AdaptiveAvgPool2d(1)` before the head, which
collapses all spatial information into a single vector. The model never "looks at"
individual pin positions — it reasons from a global texture summary.

**Solution:** Replace global pooling with 9 fixed spatial queries.  Because the crop
is always normalised to 64×64 with the diamond centred, the 9 pin positions are at
known pixel coordinates.  Extract a small patch (~12×12) centred on each position,
flatten it, and pass each through an independent tiny linear classifier (or a shared
MLP with a positional embedding).

Benefits:
- 9× more training signal per labelled crop (each pin is an independent example)
- Interpretable — you can visualise exactly which patch drives each prediction
- Far more data-efficient; should work well even with 120 crops
- The per-pin task is trivially simple: filled dark oval vs. empty light ring

Implementation sketch:
```
PIN_COORDS_64 = [
    (32, 54),               # 0  front
    (22, 42), (42, 42),     # 1  2
    (10, 32), (32, 32), (54, 32),  # 3  4  5
    (22, 22), (42, 22),     # 6  7
    (32, 10),               # 8  back
]
PATCH = 12  # pixels

class SpatialPinClassifier(nn.Module):
    def forward(self, x):          # x: (B, 1, 64, 64)
        patches = [x[:, :, y-PATCH//2:y+PATCH//2,
                         cx-PATCH//2:cx+PATCH//2]
                   for (cx, y) in PIN_COORDS_64]   # 9 × (B,1,12,12)
        feats = [self.patch_encoder(p) for p in patches]  # 9 × (B, F)
        return torch.stack([self.head(f) for f in feats], dim=1).squeeze(-1)  # (B, 9)
```

---

### A2 · Sheet-level pre-processing pipeline [ ]
**Problem:** Raw phone photos are tilted, have perspective distortion, and uneven
lighting.  YOLO receives the raw image and must generalise over all these variations.

**Solution:** Add a `preprocess_sheet` step before YOLO:
1. Detect the outer table border using contour detection (the thick black grid is
   very prominent).
2. Apply a perspective transform to produce a top-down rectified image.
3. Normalise brightness with CLAHE.

This reduces the domain that YOLO and the CNN need to cover and should meaningfully
improve both detection robustness and classification accuracy on difficult scans.

---

### A3 · Replace YOLO with a classical grid detector [ ]
**Problem:** YOLO is a heavy dependency (ultralytics) for what is essentially a
structured document — the pin diagrams sit at predictable grid positions once the
sheet is rectified (A2).

**Solution:** After perspective correction, project the rectified image onto the
vertical and horizontal axes to find rows and columns of ink density peaks.  Pair
peaks across axes to enumerate cell centres, then crop fixed-size windows.  No
learned model needed for detection at all; inference becomes deterministic.

*Note:* Keep YOLO as an optional fallback for unstructured / non-standard sheets.

---

## Domain-specific / Innovative

### D1 · OCR cross-validation of pin counts [ ]
**Problem:** The CNN may misclassify pins with no way to catch it automatically.

**Observation:** Every pin diagram on the score sheet is already annotated with the
human-verified pin count printed right next to it (e.g. "7", "1:", "0").  This is
essentially free ground truth for `sum(pins_down)`.

**Solution:** Add an OCR pass (EasyOCR or `pytesseract`) to extract the number
adjacent to each detected diagram, then compare it against `sum(predictions)`.
A mismatch is a strong signal that the CNN made an error → flag the crop for human
review or reduce its confidence score.

The trailing colon in some numbers (e.g. "1:") indicates a "Kranz" (spare); parse
this to also auto-detect Kränze.

---

### D2 · Game statistics computation [ ]
**Problem:** The pipeline currently returns only raw pin states.  The score sheet
carries richer semantics that consumers must reconstruct from scratch.

**Solution:** Add a `scoring.py` module that, given a `SheetResult`, computes:
- **Neuner** — first throw knocks all 9 pins
- **Kranz** — second throw clears all remaining pins (spare)
- **Fehlwurf** — zero pins on a throw
- **Volle total** and **Abräumen total** per lane column
- **Overall total** and per-player summary

Cross-validate derived totals against the printed totals at the bottom of the sheet
(read via OCR from D1) to provide an end-to-end consistency check.

---

### D3 · Synthetic pin diagram generation [ ]
**Problem:** Only 120 labelled crops. Even heavy augmentation cannot fully cover
the combinatorial space of 2⁹ = 512 possible pin states.

**Solution:** Render synthetic pin diagrams that match the printed style exactly:
- 9 positions in a diamond layout
- Standing pin = small empty circle / ring
- Knocked-down pin = filled dot / dash

Render all 512 states (or a weighted sample) as PNG crops at the same resolution as
real crops, with noise, rotation, and blur added.  Use as supplementary training
data.  Real crops remain the primary dataset; synthetic data is added with a lower
sample weight to prevent distribution shift.

Benefits:
- Guaranteed coverage of rare pin states
- Full 512-class label certainty
- Combined with A1 (ROI), each synthetic crop gives 9 independent training examples

---

### D4 · Volle / Abräumen column awareness [ ]
**Problem:** The pipeline treats all pin diagrams identically, but the score sheet
alternates Volle (first throw) and Abräumen (clearance) columns.  This structure
carries useful constraints:
- A Volle diagram always starts with all 9 pins standing.
- An Abräumen diagram only shows pins that were *not* knocked down in the preceding
  Volle throw.

**Solution:** Infer column parity (Volle = even columns, Abr = odd) from the
detection grid, then expose `throw_type: Literal["volle", "abr"]` in `ThrowResult`.
Use the constraint as a post-processing sanity check: if `sum(abr_pins) > 9 - sum(volle_pins)`,
something is wrong.

---

## Data Quality

### Q1 · Active learning labeler ordering [ ]
**Problem:** The labeler presents crops in alphabetical order.  The user must label
in that fixed sequence regardless of how confident the model already is.

**Solution:** When a trained classifier exists, present crops sorted by ascending
confidence (least certain first), so each labeling session targets the examples where
the model needs help most.  The labeler UI already shows confidence; just change the
default ordering.

---

### Q2 · Dataset validation command [ ]
**Problem:** Nothing checks that the labels file stays in sync with the crops on disk.
Deleted crops leave orphaned labels; new crops are silently unlabeled.

**Solution:** Add a `pinsheet-scanner validate-dataset` command that reports:
- Labeled crops whose PNG is missing
- PNG files with no label entry
- Label rows with invalid values (not 0/1, wrong length)
- Duplicate filenames in the CSV

Exit non-zero if any issues are found so it can be used in CI.

---

### Q3 · Per-pin confusion matrix in `accuracy` [ ]
**Problem:** `accuracy` reports per-pin accuracy but not the full 2×2 confusion
matrix per pin.  False-positive rate (model says "knocked down" when standing) is
very different from false-negative rate and they have different user-visible effects.

**Solution:** Add a breakdown showing `TP / FP / TN / FN` per pin position and a
summary of which specific pin is the most common source of error.

---

## Code Quality

### C1 · Extract labeler HTML to a template file [ ]
**Problem:** `cli.py` contains a 300-line raw HTML/JS string (`_LABEL_HTML`).  It
cannot be syntax-highlighted, linted, or edited with web tooling.

**Solution:** Move the template to
`src/pinsheet_scanner/labeler.html` and load it at import time with
`importlib.resources.read_text()`.  Keep the `/*CROPS_JSON*/` injection tokens.

---

### C2 · Model bundle format [ ]
**Problem:** Models are saved as bare state dicts (`torch.save(model.state_dict())`).
There is no record of which architecture version, training configuration, or data
version produced a given `.pt` file.

**Solution:** Save a bundle alongside each checkpoint:
```json
{
  "architecture": "SpatialPinClassifier",
  "input_size": [1, 64, 64],
  "num_pins": 9,
  "git_sha": "abc1234",
  "trained_at": "2026-02-19T20:00:00",
  "hyperparams": { "lr": 3e-4, "dropout": 0.3, … },
  "val_accuracy": 0.974
}
```
`load_classifier` reads the sidecar and validates that the architecture matches
before loading weights.

---

### C3 · Experiment tracking [ ]
**Problem:** Running `kfold` or `tune` multiple times produces no history.  There is
no way to know which run produced the best model or how accuracy has changed over time.

**Solution:** Append a JSONL record to `experiments.jsonl` at the end of every
`train-classifier`, `tune`, and `kfold` run:
```jsonl
{"timestamp": "…", "command": "kfold", "folds": 5, "epochs": 60,
 "git_sha": "…", "val_acc_mean": 0.974, "val_acc_std": 0.008, "hyperparams": {…}}
```
Expose a `pinsheet-scanner history` command that pretty-prints the table.

---

### C4 · Test-time augmentation (TTA) [ ]
**Problem:** Single-pass inference is sensitive to scan artifacts (skew, blur,
lighting) that the model may not have fully learned to ignore.

**Solution:** At inference time, run `K=5` randomly augmented versions of each crop
through the model and average the sigmoid probabilities before thresholding.  No
retraining required; add a `--tta` flag to `scan` and `debug-crops`.

---

### C5 · Type-check `conftest.py` / test imports [ ]
**Problem:** `test_detect.py` and `test_augment.py` import modules as bare names
(`from detect import …`) without the package prefix.  This works only because of a
`conftest.py` that patches `sys.path`, which is fragile.

**Solution:** Migrate all test imports to the full package path
(`from pinsheet_scanner.detect import …`) and remove any `sys.path` hacks.

---

## Done

- [x] Pin numbering convention reversed — front = index 0 (2026-02-19)
