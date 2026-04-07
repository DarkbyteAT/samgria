# Contributing to samgria

## Development Setup

```bash
git clone https://github.com/DarkbyteAT/samgria.git
cd samgria
source scripts/enable-venv.sh
```

## Code Conventions

- **Python 3.11+** — `X | Y` union syntax, `list[T]`/`dict[K,V]` generics
- **Google-style docstrings**

## Docstring Style

Google-style docstrings with LaTeX math support:

```python
"""Compute the SAM perturbation $\epsilon^* \approx \rho \nabla L / \|\nabla L\|$.

Args:
    rho: Perturbation radius $\rho$.
    grad: Current gradient $\nabla L(\theta)$.

Returns:
    Adversarial perturbation vector.
"""
```

Use `$...$` for inline math and `$$...$$` for display math in docstrings.

## Quality Gates

```bash
make all    # format-check + lint + typecheck + test
make fix    # auto-fix lint violations
```

Tool configs live in dedicated files (`ruff.toml`, `pytest.ini`, `pyrightconfig.json`).

## Testing

- Plain `def test_*` functions — no classes
- Given-When-Then structure
- `tests/` mirrors `samgria/` layout
