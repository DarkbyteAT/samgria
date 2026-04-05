.PHONY: lint format format-check fix typecheck test all

lint:
	uv run ruff check PACKAGE_NAME/

format:
	uv run ruff format PACKAGE_NAME/

format-check:
	uv run ruff format --check PACKAGE_NAME/

fix:
	uv run ruff check --fix PACKAGE_NAME/

typecheck:
	uv run pyright PACKAGE_NAME/

test:
	uv run pytest tests/ -v || [ $$? -eq 5 ]

all: format-check lint typecheck test
