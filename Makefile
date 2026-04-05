.PHONY: lint format format-check fix typecheck test all

lint:
	uv run ruff check samgria/

format:
	uv run ruff format samgria/

format-check:
	uv run ruff format --check samgria/

fix:
	uv run ruff check --fix samgria/

typecheck:
	uv run pyright samgria/

test:
	uv run pytest tests/ -v || [ $$? -eq 5 ]

all: format-check lint typecheck test
