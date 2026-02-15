# Pinsheet Scanner — task runner
# Run `just` or `just --list` to see available tasks.

default:
    @just --list

# ── Setup ─────────────────────────────────────────────────────────

# Install all dependencies (including dev)
install:
    uv sync --all-extras

# ── Quality ───────────────────────────────────────────────────────

# Run unit tests
test:
    uv run pytest -q

# Run integration tests (requires model weights)
integration:
    uv run pytest -q -m integration

# Lint and format with ruff
lint:
    ruff check . --fix && ruff format .

# Type-check sources
typecheck:
    basedpyright src

# ── Synthetic Data & Classifier Training ──────────────────────────

# Delete stale synthetic data
clean-data:
    rm -rf data/classifier

# Generate synthetic pin diagram training data
generate-data *args:
    uv run python -m scripts.generate_data {{ args }}

# Train the CNN pin classifier on synthetic data
train-classifier *args:
    uv run python -m scripts.train_classifier {{ args }}

# Regenerate data from scratch and retrain (clean → generate → train)
retrain: clean-data generate-data train-classifier

# ── YOLO Detector Training ────────────────────────────────────────

# Train the YOLO detector for pin diagram bounding boxes
train-detector *args:
    uv run python -m scripts.train {{ args }}

# ── Inference ─────────────────────────────────────────────────────

# Scan a score sheet and print results
scan image *args:
    uv run pinsheet-scanner {{ image }} {{ args }}

# Save debug crops from a score sheet image
debug-crops image *args:
    uv run python -m scripts.debug_crops {{ image }} {{ args }}

# ── Validation ────────────────────────────────────────────────────

# Validate the CNN classifier on real crops in debug_crops/raw/
validate:
    #!/usr/bin/env bash
    set -euo pipefail
    crop_dir="debug_crops/raw"
    if [ ! -d "$crop_dir" ]; then
        echo "No crops found at $crop_dir — run 'just debug-crops <image>' first."
        exit 1
    fi
    count=$(find "$crop_dir" -name '*.png' | wc -l | tr -d ' ')
    echo "Validating classifier on $count crops in $crop_dir …"
    uv run python -c "
    from pathlib import Path
    import cv2
    from pinsheet_scanner.classify import load_classifier, classify_pins_with_confidence

    crop_dir = Path('$crop_dir')
    model, device = load_classifier(Path('models/pin_classifier.pt'))

    total, confident, low_conf = 0, 0, []
    for p in sorted(crop_dir.glob('*.png')):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pins, conf = classify_pins_with_confidence(model, img, device=device)
        score = sum(pins)
        total += 1
        if conf >= 0.8:
            confident += 1
        else:
            low_conf.append((p.stem, conf))
        pin_str = ''.join(str(x) for x in pins)
        print(f'{p.stem:>20}  pins={pin_str}  score={score}  conf={conf:.2f}')

    print(f'\n{total} crops | {confident} high-confidence (≥0.80) | {total - confident} low-confidence')
    if low_conf:
        print('Low-confidence crops:')
        for name, c in sorted(low_conf, key=lambda x: x[1]):
            print(f'  {name}: {c:.2f}')
    "

# ── Cleanup (post-migration) ─────────────────────────────────────

# Remove obsolete pre-migration files (classical CV classifiers, OCR, debug dirs)
clean-legacy:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Removing obsolete files …"
    rm -fv src/pinsheet_scanner/classify_template.py
    rm -fv src/pinsheet_scanner/classify_contour.py
    rm -fv src/pinsheet_scanner/ocr.py
    rm -fv tests/test_classify_template.py
    rm -fv tests/test_ocr.py
    rm -rfv debug_crops/cleaned/
    rm -rfv debug_crops/resized/
    rm -rfv debug_crops/overlay/
    echo "Checking for stale imports …"
    if grep -rn "classify_template\|classify_contour\|ocr\|pytesseract\|pins_from_diagram\|DEFAULT_MASK_RADIUS" src/ tests/ 2>/dev/null; then
        echo "⚠  Found stale references — clean these up manually."
    else
        echo "✓  No stale references found."
    fi
