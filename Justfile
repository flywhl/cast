default:
    @just --list

test:
    @uv run pytest

test-s:
    @uv run pytest -s -o log_cli=True -o log_cli_level=DEBUG

ruff:
    uv run ruff format cyantic

pyright:
    uv run pyright cyantic

lint:
    just ruff
    just pyright

lint-file file:
    - ruff {{file}}
    - pyright {{file}}
