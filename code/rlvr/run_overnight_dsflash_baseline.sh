#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TOKENPLAN_API_KEY:-}" ]]; then
  echo "TOKENPLAN_API_KEY is required" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

RUN_ID="${RUN_ID:-overnight_dsflash_20260914}"
RUN_DIR="artifacts/${RUN_ID}"
MODEL_PATH="${MODEL_PATH:-/home/gpuuser/csl/models/Qwen3-4B-Instruct-2507}"
JUDGE_MODEL="deepseek-v4-flash-0731"
SEED="20260914"

mkdir -p "$RUN_DIR"

echo "[$(date -u +%FT%TZ)] start run_id=$RUN_ID"
echo "[$(date -u +%FT%TZ)] generation: official 42 x {no_skill,skill_text}, seed=$SEED"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" uv run python code/rlvr/run_local_baseline.py \
  --model "$MODEL_PATH" \
  --tasks data/single_turn_tasks.csv \
  --output "$RUN_DIR/rollouts.jsonl" \
  --condition both \
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed "$SEED"

echo "[$(date -u +%FT%TZ)] generation complete"

complete=0
for pass in $(seq 1 12); do
  echo "[$(date -u +%FT%TZ)] judge pass=$pass model=$JUDGE_MODEL"
  if ! uv run python code/rlvr/score_rollouts.py \
    --input "$RUN_DIR/rollouts.jsonl" \
    --output "$RUN_DIR/scores.jsonl" \
    --tasks data/single_turn_tasks.csv \
    --judge-model "$JUDGE_MODEL" \
    --judge-protocol paper_pass_fail \
    --max-tokens 2048 \
    --max-tokens-cap 8192 \
    --concurrency 2; then
    echo "[$(date -u +%FT%TZ)] judge pass=$pass transport/preflight failure; retrying" >&2
    sleep 30
    continue
  fi

  if uv run python code/rlvr/summarize_baseline_scores.py \
    --input "$RUN_DIR/scores.jsonl" \
    --output-dir "$RUN_DIR/summary" \
    --judge-model "$JUDGE_MODEL" \
    --seed "$SEED" \
    --require-complete; then
    complete=1
    break
  fi
  echo "[$(date -u +%FT%TZ)] incomplete after judge pass=$pass; retrying missing cells in 30s"
  sleep 30
done

uv run python code/rlvr/summarize_baseline_scores.py \
  --input "$RUN_DIR/scores.jsonl" \
  --output-dir "$RUN_DIR/summary" \
  --judge-model "$JUDGE_MODEL" \
  --seed "$SEED"

if [[ "$complete" -ne 1 ]]; then
  echo "[$(date -u +%FT%TZ)] finished with missing Judge cells; inspect summary/overall.json" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] complete"
