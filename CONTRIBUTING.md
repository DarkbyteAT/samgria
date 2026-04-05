# Contributing to PACKAGE_NAME

## Development Setup

```bash
git clone https://github.com/DarkbyteAT/PACKAGE_NAME.git
cd PACKAGE_NAME
source scripts/enable-venv.sh
```

## Code Conventions

- **Python 3.11+** — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- **NumPy-style docstrings**

## Quality Gates

```bash
make all    # format-check + lint + typecheck + test
make fix    # auto-fix lint violations
```

Tool configs live in dedicated files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`).

## Testing

- Plain `def test_*` functions — no classes
- Given-When-Then structure
- `tests/` mirrors `PACKAGE_NAME/` layout
