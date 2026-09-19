#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SERVE_ENV="${QWEN4B_SERVE_ENV:-$REPO_ROOT/.venv-qwen4b-serve}"

if [ ! -x "$SERVE_ENV/bin/python" ]; then
  uv venv --python 3.12 "$SERVE_ENV"
fi
uv pip install --python "$SERVE_ENV/bin/python" \
  'vllm>=0.8.5,<0.9' \
  'transformers>=4.51,<5'
"$SERVE_ENV/bin/vllm" --version
echo "Serving environment ready: $SERVE_ENV"
