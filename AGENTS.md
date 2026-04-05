# AGENTS.md

## Commands

```bash
source scripts/enable-venv.sh
uv run ruff check samgria/
uv run pyright samgria/
uv run pytest tests/
make all
```

## Critical Rules

- Transforms implement the GradientTransform protocol (apply + post_step)
- Protocol uses nn.Module (not nn.ModuleDict) for generality
- Python 3.11+, PyTorch aliases: `T` for `torch`, `nn` for `torch.nn`, `F` for `torch.nn.functional`
- Plain `def test_*` functions, Given-When-Then structure
