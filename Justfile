default:
    @just --list

test:
    @pytest

test-s:
    @pytest -s

ruff:
    uv run ruff format cast

pyright:
    uv run pyright cast

lint:
    just ruff
    just pyright

lint-file file:
    - ruff {{file}}
    - pyright {{file}}

