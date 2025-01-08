default:
    @just --list

test:
    @uv run pytest

test-s:
    @uv run pytest -s -o log_cli=True -o log_cli_level=DEBUG

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

