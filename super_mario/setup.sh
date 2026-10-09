#!/usr/bin/env bash
# Creates an isolated Python 3.10 environment in .venv and installs dependencies.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found - install it with: brew install uv" >&2
  exit 1
fi

uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then check the game:  python test_env.py"
