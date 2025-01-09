default:
    @just --list

test:
    @uv run pytest

test-s:
    @uv run pytest -s -o log_cli=True -o log_cli_level=DEBUG

ruff-fix:
    uv run ruff format cyantic

ruff-check:
    uv run ruff check cyantic

pyright:
    uv run pyright cyantic

lint:
    just ruff-check
    just pyright

lint-file file:
    - ruff {{file}}
    - pyright {{file}}
