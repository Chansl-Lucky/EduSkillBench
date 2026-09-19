#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_DIR="artifacts/qwen4b_skill_baseline_rescore_001"
MODEL_PATH="${QWEN3_4B_MODEL_PATH:-${QWEN4B_MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "QWEN3_4B_MODEL_PATH is required" >&2
  exit 2
fi
JUDGE_MODEL="qwen3-4b-instruct-2507-cdbee75-local"
mkdir -p "$RUN_DIR/summary"

echo "[$(date -u +%FT%TZ)] start Qwen Judge rescore of frozen 84-cell Skill baseline"
for pass in 1 2 3; do
  echo "[$(date -u +%FT%TZ)] pass=$pass"
  CUDA_VISIBLE_DEVICES=1 uv run python code/rlvr/score_rollouts_local_qwen.py \
    --input artifacts/overnight_dsflash_20260914/rollouts.jsonl \
    --output "$RUN_DIR/scores.jsonl" \
    --tasks data/single_turn_tasks.csv \
    --model "$MODEL_PATH" \
    --judge-model "$JUDGE_MODEL" \
    --max-new-tokens 1024 \
    --max-new-tokens-cap 2048

  if uv run python code/rlvr/summarize_baseline_scores.py \
    --input "$RUN_DIR/scores.jsonl" \
    --output-dir "$RUN_DIR/summary" \
    --tasks data/single_turn_tasks.csv \
    --judge-model "$JUDGE_MODEL" \
    --seed 20260914 \
    --require-complete; then
    echo "[$(date -u +%FT%TZ)] complete"
    exit 0
  fi
done

echo "[$(date -u +%FT%TZ)] finished incomplete" >&2
exit 1
