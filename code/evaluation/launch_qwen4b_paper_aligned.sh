#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

if tmux has-session -t eduskill-feasibility002 2>/dev/null; then
  echo "Current reward_feasibility_002 is still active; launch postponed" >&2
  exit 1
fi
if tmux has-session -t eduskill-qwen4b-server 2>/dev/null || \
   tmux has-session -t eduskill-qwen4b-eval 2>/dev/null; then
  echo "A Qwen4B server/eval tmux session already exists" >&2
  exit 1
fi

code/evaluation/preflight_qwen4b_paper_aligned.sh --static
DOCKER_GATEWAY=$(docker network inspect bridge \
  --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')

mkdir -p logs
tmux new-session -d -s eduskill-qwen4b-server \
  "cd '$REPO_ROOT' && exec code/evaluation/start_qwen4b_server.sh >> logs/qwen3_4b_vllm.log 2>&1"

for _ in $(seq 1 120); do
  if curl -fsS -H "Authorization: Bearer ${LOCAL_QWEN_API_KEY:-local-eduskillbench}" \
    "http://$DOCKER_GATEWAY:${QWEN4B_PORT:-8000}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

uv run python code/evaluation/smoke_qwen4b_api.py
tmux new-session -d -s eduskill-qwen4b-eval \
  "cd '$REPO_ROOT' && exec code/evaluation/run_qwen4b_paper_aligned.sh"

echo "Started: eduskill-qwen4b-server and eduskill-qwen4b-eval"
echo "Logs: logs/qwen3_4b_vllm.log and logs/paper_aligned_qwen3_4b_001.log"
