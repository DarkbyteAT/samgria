# Tests

## Philosophy

Test behaviour, not implementation. Tests prove the code does what it should, catch regressions, and serve as executable examples of correct usage. Follow the Given-When-Then pattern for readability.

## Organisation

The test structure mirrors the source layout. Each subdirectory under `tests/` corresponds to a module in the source tree. When adding tests for a new module, create a matching subdirectory under `tests/`.

Shared test helpers and fixtures live in `tests/conftest.py` (global) and per-directory `conftest.py` files for scoped fixtures.

## Test Markers

Every test must be marked with exactly one of:

| Marker | When to use | What it tests |
|--------|------------|---------------|
| `@pytest.mark.unit` | Default for most tests | A single function or class in isolation, mocked dependencies |
| `@pytest.mark.integration` | Tests that cross module boundaries | Multiple components wired together, real dependencies |
| `@pytest.mark.e2e` | Full pipeline tests | End-to-end workflows matching real usage patterns |

Run by marker:

```bash
pytest -m unit          # fast, isolated — runs in pre-commit hooks
pytest -m integration   # cross-module, may be slower
pytest -m e2e           # full pipeline, slowest
pytest                  # all tests
```

Pre-commit hooks run `pytest -m unit` only. CI runs all markers.

## Conventions

- Plain `def test_*` functions — no test classes
- Given-When-Then structure for all tests
- Helper modules prefixed with `_` to distinguish from test files
- Mark every test — unmarked tests won't run in pre-commit

## Running Tests

```bash
source scripts/enable-venv.sh
pytest tests/           # all tests
pytest tests/ -m unit   # unit only (fast)
pytest tests/ -q        # quiet output
```

All tests must pass before committing. Pre-commit hooks enforce unit tests.
