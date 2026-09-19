#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"
MODE="${1:---full}"
case "$MODE" in
  --full|--static) ;;
  *) echo "usage: $0 [--full|--static]" >&2; exit 2 ;;
esac

fail=0
check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    echo "OK      command $1"
  else
    echo "MISSING command $1"
    fail=1
  fi
}

check_cmd uv
check_cmd docker
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "OK      Docker daemon"
  else
    echo "MISSING Docker daemon access"
    fail=1
  fi
fi

if [ -x .venv-qwen4b-serve/bin/vllm ]; then
  echo "OK      isolated vLLM environment"
else
  echo "MISSING .venv-qwen4b-serve (run setup_qwen4b_serving_env.sh)"
  fail=1
fi

MODEL_PATH="${QWEN4B_MODEL_PATH:-${QWEN3_4B_MODEL_PATH:-}}"
if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
  echo "OK      pinned local model directory"
else
  echo "MISSING local model directory (set QWEN4B_MODEL_PATH or QWEN3_4B_MODEL_PATH)"
  fail=1
fi

if [ -f artifacts/paper_aligned_qwen3_4b_001/input_manifest.json ]; then
  echo "OK      prepared 42-case input manifest"
else
  echo "MISSING prepared input manifest"
  fail=1
fi

if code/utils/patch_benchflow.sh --check >/dev/null 2>&1; then
  echo "OK      BenchFlow local-judge routing patch"
else
  echo "MISSING BenchFlow local-judge routing patch"
  fail=1
fi

if [ "$MODE" = "--full" ]; then
  DOCKER_GATEWAY=$(docker network inspect bridge \
    --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || true)
  if [ -n "$DOCKER_GATEWAY" ] && \
    curl -fsS -H "Authorization: Bearer ${LOCAL_QWEN_API_KEY:-local-eduskillbench}" \
      "http://$DOCKER_GATEWAY:${QWEN4B_PORT:-8000}/v1/models" >/dev/null 2>&1; then
    echo "OK      local Qwen API"
  else
    echo "WAIT    local Qwen API is not running (expected before launch)"
    fail=1
  fi
fi

exit "$fail"
