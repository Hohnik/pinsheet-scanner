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
    ruff check . --fix && ruff format .

# K-fold cross-validate then retrain the CNN classifier
train *args:
    uv run pinsheet-scanner train {{ args }}

# Hyperparameter tuning with Optuna
tune *args:
    uv run pinsheet-scanner tune {{ args }}

# Train the YOLO detector for pin diagram bounding boxes
train-detector *args:
    uv run pinsheet-scanner train-detector {{ args }}

# Scan a score sheet and print results
scan image *args:
    uv run pinsheet-scanner scan {{ image }} {{ args }}

# Extract crops from a score sheet image
extract image *args:
    uv run pinsheet-scanner extract {{ image }} {{ args }}

# Open the labeling UI to annotate ground-truth pin states
label *args:
    uv run pinsheet-scanner label {{ args }}

# Compare ground-truth labels against CNN predictions
accuracy *args:
    uv run pinsheet-scanner accuracy {{ args }}
