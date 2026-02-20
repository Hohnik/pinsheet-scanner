_default:
    @just --list --unsorted

# Install all dependencies (including dev)
install:
    uv sync --all-extras

# Run unit tests
test:
    uv run pytest -q

# Run integration tests (requires model weights)
integration:
    uv run pytest -q -m integration

# Lint and format with ruff
lint:
    uv run ruff check . --fix && uv run ruff format .

# K-fold cross-validate then retrain the CNN classifier
train *args:
    uv run pinsheet-scanner train {{ args }}

# Hyperparameter tuning with Optuna
tune *args:
    uv run pinsheet-scanner tune {{ args }}

# Train the YOLO detector for pin diagram bounding boxes
train-detector *args:
    uv run pinsheet-scanner train-detector {{ args }}

# Scan one or more score sheets and print results
scan *args:
    uv run pinsheet-scanner scan {{ args }}

# Harvest high-confidence crops from sheet(s) into the training set
collect *args="sheets/*":
    uv run pinsheet-scanner collect {{ args }}

# Extract crops from one or more score sheet images
extract *args="sheets/*":
    uv run pinsheet-scanner extract {{ args }}

# Open the labeling UI to annotate ground-truth pin states
label *args:
    uv run pinsheet-scanner label {{ args }}

# Compare ground-truth labels against CNN predictions
accuracy *args:
    uv run pinsheet-scanner accuracy {{ args }}

# Profile the scan pipeline (cProfile + pyinstrument flame graph)
profile:
    uv run python scripts/profile_scan.py
