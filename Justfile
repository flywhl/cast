default:
    @just --list

test:
    @pytest

test-s:
    @pytest -s

ruff:
    rye run ruff format cast

pyright:
    rye run pyright cast

lint:
    just ruff
    just pyright

lint-file file:
    - ruff {{file}}
    - pyright {{file}}

