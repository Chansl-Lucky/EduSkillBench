#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

uv sync --extra train --extra dev
echo "Environment ready. Copy .env.example to .env and export the variables before running."
