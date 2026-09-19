#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SERVE_ENV="${QWEN4B_SERVE_ENV:-$REPO_ROOT/.venv-qwen4b-serve}"
MODEL_PATH="${QWEN4B_MODEL_PATH:-/home/gpuuser/csl/models/Qwen3-4B-Instruct-2507}"
SERVED_NAME="${QWEN4B_SERVED_NAME:-qwen3-4b-instruct-2507-cdbee75}"
PORT="${QWEN4B_PORT:-8000}"
GPU="${QWEN4B_GPU:-1}"
API_KEY="${LOCAL_QWEN_API_KEY:-local-eduskillbench}"
HOST="${QWEN4B_HOST:-$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')}"

if [ ! -x "$SERVE_ENV/bin/vllm" ]; then
  echo "Missing $SERVE_ENV/bin/vllm; run code/evaluation/setup_qwen4b_serving_env.sh first" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "Missing model directory: $MODEL_PATH" >&2
  exit 1
fi
if [ -z "$HOST" ]; then
  echo "Could not determine Docker bridge gateway" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
exec "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
