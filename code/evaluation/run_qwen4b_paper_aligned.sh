#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

INPUT_ROOT="artifacts/paper_aligned_qwen3_4b_001/input_skills"
JOBS_ROOT="jobs/paper-aligned-qwen3-4b-001"
MODEL="vllm/qwen3-4b-instruct-2507-cdbee75"
PORT="${QWEN4B_PORT:-8000}"
API_KEY="${LOCAL_QWEN_API_KEY:-local-eduskillbench}"

if tmux has-session -t eduskill-feasibility002 2>/dev/null; then
  echo "Refusing to overlap reward_feasibility_002; wait for that tmux session to end" >&2
  exit 1
fi
if [ ! -f "$INPUT_ROOT/../input_manifest.json" ]; then
  echo "Prepared inputs missing; run prepare_qwen4b_paper_aligned.py first" >&2
  exit 1
fi
code/utils/patch_benchflow.sh --check
docker info >/dev/null
DOCKER_GATEWAY=$(docker network inspect bridge \
  --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')
if [ -z "$DOCKER_GATEWAY" ]; then
  echo "Could not determine Docker bridge gateway" >&2
  exit 1
fi
curl -fsS -H "Authorization: Bearer $API_KEY" \
  "http://$DOCKER_GATEWAY:$PORT/v1/models" >/dev/null
uv run python code/evaluation/smoke_qwen4b_api.py

# Policy traffic: BenchFlow host LiteLLM -> bridge-bound vLLM.
export BENCHFLOW_PROVIDER_BASE_URL="http://$DOCKER_GATEWAY:$PORT/v1"
export BENCHFLOW_PROVIDER_API_KEY="$API_KEY"
# Primary judge executes inside Docker and reaches that same vLLM instance via
# the bridge gateway.  The patched generator forwards this endpoint.
export OPENAI_BASE_URL="http://$DOCKER_GATEWAY:$PORT/v1"
export OPENAI_API_KEY="$API_KEY"

mkdir -p logs "$JOBS_ROOT"
for skill_dir in "$INPUT_ROOT"/*; do
  [ -f "$skill_dir/evals/evals.json" ] || continue
  skill=$(basename "$skill_dir")
  echo "========== $skill =========="
  uv run bench skills eval "$skill_dir" \
    --agent opencode \
    --model "$MODEL" \
    --sandbox docker \
    --concurrency 1 \
    --jobs-dir "$JOBS_ROOT/$skill"
done 2>&1 | tee "logs/paper_aligned_qwen3_4b_001.log"

uv run python code/evaluation/summarize_qwen4b_paper_aligned.py
