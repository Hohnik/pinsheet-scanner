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
train *args:
    uv run pinsheet-scanner train-classifier {{ args }}

# Hyperparameter tuning with Optuna
tune *args:
    uv run pinsheet-scanner tune {{ args }}

# K-fold cross-validation for the CNN classifier
kfold *args:
    uv run pinsheet-scanner kfold {{ args }}

# Train the YOLO detector for pin diagram bounding boxes
train-detector *args:
    uv run pinsheet-scanner train-detector {{ args }}

# ── Inference ─────────────────────────────────────────────────────

# Scan a score sheet and print results
scan image *args:
    uv run pinsheet-scanner scan {{ image }} {{ args }}

# Extract crops from a score sheet image
extract image *args:
    uv run pinsheet-scanner debug-crops {{ image }} {{ args }}

# ── Labeling & Validation ────────────────────────────────────────

# Open the labeling UI to annotate ground-truth pin states
label *args:
    uv run pinsheet-scanner label {{ args }}

# Compare ground-truth labels against CNN predictions
accuracy *args:
    uv run pinsheet-scanner accuracy {{ args }}
