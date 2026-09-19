#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TOKENPLAN_API_KEY:-}" ]]; then
  echo "TOKENPLAN_API_KEY is required" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

SOURCE_DIR="artifacts/reward_feasibility_001"
RUN_DIR="artifacts/reward_feasibility_002"
TASKS="$SOURCE_DIR/prompts.jsonl"
ROLLOUTS="$SOURCE_DIR/rollouts.jsonl"
AUDIT_SCORES="$SOURCE_DIR/scores.jsonl"
JUDGE_MODEL="deepseek-v4-flash-0731"
SEEDS=(20260915 20260916 20260917 20260918)

mkdir -p "$RUN_DIR/summary"
[[ $(wc -l < "$TASKS") -eq 42 ]] || { echo "expected 42 frozen prompts" >&2; exit 1; }
[[ $(wc -l < "$ROLLOUTS") -eq 168 ]] || { echo "expected 168 frozen rollouts" >&2; exit 1; }

echo "[$(date -u +%FT%TZ)] start run_id=reward_feasibility_002"
echo "[$(date -u +%FT%TZ)] reuse frozen prompts=42 rollouts=168; paper audit remains unchanged"

complete=0
for pass in $(seq 1 12); do
  echo "[$(date -u +%FT%TZ)] dense judge pass=$pass model=$JUDGE_MODEL"
  if ! uv run python code/rlvr/score_rollouts.py \
    --input "$ROLLOUTS" \
    --output "$RUN_DIR/dense_scores.jsonl" \
    --tasks "$TASKS" \
    --judge-model "$JUDGE_MODEL" \
    --judge-protocol rubric_continuous \
    --max-tokens 2048 \
    --max-tokens-cap 8192 \
    --concurrency 1; then
    echo "[$(date -u +%FT%TZ)] dense judge transport/preflight failure; retrying in 30s" >&2
    sleep 30
    continue
  fi

  if uv run python code/rlvr/summarize_reward_feasibility.py \
    --scores "$RUN_DIR/dense_scores.jsonl" \
    --rollouts "$ROLLOUTS" \
    --tasks "$TASKS" \
    --output-dir "$RUN_DIR/summary" \
    --judge-model "$JUDGE_MODEL" \
    --judge-protocol rubric_continuous \
    --seeds "${SEEDS[@]}" \
    --require-complete; then
    complete=1
    break
  fi
  echo "[$(date -u +%FT%TZ)] incomplete after dense judge pass=$pass; retrying in 30s"
  sleep 30
done

uv run python code/rlvr/summarize_reward_feasibility.py \
  --scores "$RUN_DIR/dense_scores.jsonl" \
  --rollouts "$ROLLOUTS" \
  --tasks "$TASKS" \
  --output-dir "$RUN_DIR/summary" \
  --judge-model "$JUDGE_MODEL" \
  --judge-protocol rubric_continuous \
  --seeds "${SEEDS[@]}"

uv run python code/rlvr/compare_reward_protocols.py \
  --dense-scores "$RUN_DIR/dense_scores.jsonl" \
  --audit-scores "$AUDIT_SCORES" \
  --output-dir "$RUN_DIR/summary"

if [[ "$complete" -ne 1 ]]; then
  echo "[$(date -u +%FT%TZ)] finished incomplete" >&2
  exit 1
fi
echo "[$(date -u +%FT%TZ)] complete"
