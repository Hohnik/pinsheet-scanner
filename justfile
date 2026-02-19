_default:
    @just --list --unsorted

# Install all dependencies (including dev)
[group('setup')]
install:
    uv sync --all-extras

# ── Quality ───────────────────────────────────────────────────────────────

# Run unit tests
[group('quality')]
test:
    uv run pytest -q

# Run integration tests (requires model weights)
[group('quality')]
integration:
    uv run pytest -q -m integration

# Lint and format with ruff
[group('quality')]
lint:
    ruff check . --fix && ruff format .

# Type-check sources
[group('quality')]
typecheck:
    basedpyright src

# ── Training ──────────────────────────────────────────────────────────────

# K-fold cross-validate then retrain the CNN classifier
[group('training')]
train *args:
    uv run pinsheet-scanner train {{ args }}

# Hyperparameter tuning with Optuna
[group('training')]
tune *args:
    uv run pinsheet-scanner tune {{ args }}

# Train the YOLO detector for pin diagram bounding boxes
[group('training')]
train-detector *args:
    uv run pinsheet-scanner train-detector {{ args }}

# ── Inference ─────────────────────────────────────────────────────────────

# Scan a score sheet and print results
[group('inference')]
scan image *args:
    uv run pinsheet-scanner scan {{ image }} {{ args }}

# Extract crops from a score sheet image
[group('inference')]
extract image *args:
    uv run pinsheet-scanner extract {{ image }} {{ args }}

# ── Labeling & Validation ─────────────────────────────────────────────────

# Open the labeling UI to annotate ground-truth pin states
[group('labeling')]
label *args:
    uv run pinsheet-scanner label {{ args }}

# Compare ground-truth labels against CNN predictions
[group('validation')]
accuracy *args:
    uv run pinsheet-scanner accuracy {{ args }}
