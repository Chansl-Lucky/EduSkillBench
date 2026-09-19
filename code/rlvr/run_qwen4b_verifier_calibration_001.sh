#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_DIR="artifacts/qwen4b_verifier_calibration_001"
MODEL_PATH="${QWEN3_4B_MODEL_PATH:-${QWEN4B_MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "QWEN3_4B_MODEL_PATH is required" >&2
  exit 2
fi
JUDGE_MODEL="qwen3-4b-instruct-2507-cdbee75-local"
SEEDS=(20260915 20260916 20260917 20260918)
mkdir -p "$RUN_DIR/summary"

echo "[$(date -u +%FT%TZ)] start qwen4b local verifier calibration"
for pass in 1 2 3; do
  echo "[$(date -u +%FT%TZ)] pass=$pass"
  CUDA_VISIBLE_DEVICES=1 uv run python code/rlvr/score_rollouts_local_qwen.py \
    --input artifacts/reward_feasibility_001/rollouts.jsonl \
    --output "$RUN_DIR/scores.jsonl" \
    --tasks artifacts/reward_feasibility_001/prompts.jsonl \
    --model "$MODEL_PATH" \
    --judge-model "$JUDGE_MODEL" \
    --max-new-tokens 1024 \
    --max-new-tokens-cap 2048

  if uv run python code/rlvr/summarize_reward_feasibility.py \
    --scores "$RUN_DIR/scores.jsonl" \
    --rollouts artifacts/reward_feasibility_001/rollouts.jsonl \
    --tasks artifacts/reward_feasibility_001/prompts.jsonl \
    --output-dir "$RUN_DIR/summary" \
    --judge-model "$JUDGE_MODEL" \
    --judge-protocol rubric_continuous \
    --seeds "${SEEDS[@]}" \
    --require-complete; then
    echo "[$(date -u +%FT%TZ)] complete"
    exit 0
  fi
done

uv run python code/rlvr/summarize_reward_feasibility.py \
  --scores "$RUN_DIR/scores.jsonl" \
  --rollouts artifacts/reward_feasibility_001/rollouts.jsonl \
  --tasks artifacts/reward_feasibility_001/prompts.jsonl \
  --output-dir "$RUN_DIR/summary" \
  --judge-model "$JUDGE_MODEL" \
  --judge-protocol rubric_continuous \
  --seeds "${SEEDS[@]}"
echo "[$(date -u +%FT%TZ)] finished incomplete" >&2
exit 1
