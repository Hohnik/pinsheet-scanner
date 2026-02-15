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

# ── Training ─────────────────────────────────────────────────────

# Train the CNN pin classifier on real labeled crops
train-classifier *args:
    uv run python -m scripts.train_classifier {{ args }}

# Retrain from scratch on real crops
retrain:
    uv run python -m scripts.train_classifier

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

# ── Labeling & Validation ────────────────────────────────────────

# Open the labeling UI to annotate ground-truth pin states
label *args:
    uv run python -m scripts.label {{ args }}

# Compare ground-truth labels against CNN predictions
accuracy *args:
    uv run python -m scripts.validate_labels {{ args }}

# Run CNN on real crops and print per-crop predictions + confidence
validate:
    uv run python -m scripts.debug_crops pinsheet_example.jpeg --output debug_crops
