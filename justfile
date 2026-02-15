default:
    @just --list

install:
    uv sync

test:
    uv run pytest -q

integration:
    uv run pytest -q -m integration

lint:
    ruff check . --fix && ruff format .

typecheck:
    basedpyright src

scan image:
    uv run pinsheet-scanner {{image}}

scan-model image model:
    uv run pinsheet-scanner {{image}} --model {{model}}

scan-confident image confidence="0.5":
    uv run pinsheet-scanner {{image}} --confidence {{confidence}}

debug-crops image:
    uv run python -m scripts.debug_crops {{image}}

train:
    uv run python -m scripts.train
