#!/usr/bin/env bash
VENV_DIR=".venv"
if [ -n "$VIRTUAL_ENV" ]; then return; fi
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at '$VENV_DIR'..."
  python -m venv "$VENV_DIR" || { echo "venv creation failed"; return 1; }
fi
echo "Activating venv..."
source "$VENV_DIR/bin/activate" || { echo "venv activation failed"; return 1; }
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
  uv sync --group dev || { echo "uv sync failed"; return 1; }
else
  pip install --upgrade pip && pip install -e ".[dev]" || { echo "pip install failed"; return 1; }
fi
if [ ! -f ".git/hooks/pre-commit" ]; then
  echo "Installing pre-commit hooks..."
  pre-commit install || { echo "pre-commit install failed"; return 1; }
fi
