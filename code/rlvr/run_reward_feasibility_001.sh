#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TOKENPLAN_API_KEY:-}" ]]; then
  echo "TOKENPLAN_API_KEY is required" >&2
  exit 2
fi

cd "$(dirname "$0")/../.."

RUN_ID="reward_feasibility_001"
RUN_DIR="artifacts/$RUN_ID"
POOL_DIR="$RUN_DIR/pools"
MODEL_PATH="${MODEL_PATH:-/home/gpuuser/csl/models/Qwen3-4B-Instruct-2507}"
GENERATOR_MODEL="Atria-Dawn-Preview"
JUDGE_MODEL="deepseek-v4-flash-0731"
SEEDS=(20260915 20260916 20260917 20260918)

mkdir -p "$POOL_DIR" "$RUN_DIR/summary"
echo "[$(date -u +%FT%TZ)] start run_id=$RUN_ID"

for skill_dir in skills/single_turn/*; do
  [[ -d "$skill_dir" ]] || continue
  skill_id=$(basename "$skill_dir")
  output="$POOL_DIR/$skill_id.jsonl"
  if [[ -s "$output" ]]; then
    echo "[$(date -u +%FT%TZ)] prompt pool exists skill=$skill_id"
    continue
  fi
  generated=0
  for attempt in $(seq 1 6); do
    echo "[$(date -u +%FT%TZ)] generate skill=$skill_id attempt=$attempt"
    if uv run python code/rlvr/generate_prompt_pool.py \
      --skill-id "$skill_id" \
      --split judge_calibration \
      --count 3 \
      --model "$GENERATOR_MODEL" \
      --max-tokens 8192 \
      --output "$output"; then
      generated=1
      break
    fi
    sleep 30
  done
  if [[ "$generated" -ne 1 ]]; then
    echo "failed to generate prompt pool for $skill_id" >&2
    exit 1
  fi
done

uv run python code/rlvr/assemble_prompt_pools.py \
  --input-dir "$POOL_DIR" \
  --output "$RUN_DIR/prompts.jsonl" \
  --report "$RUN_DIR/prompt_audit.json"
echo "[$(date -u +%FT%TZ)] prompt pool complete"

for seed in "${SEEDS[@]}"; do
  echo "[$(date -u +%FT%TZ)] rollout seed=$seed"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" uv run python \
    code/rlvr/run_local_baseline.py \
    --model "$MODEL_PATH" \
    --tasks "$RUN_DIR/prompts.jsonl" \
    --output "$RUN_DIR/rollouts.jsonl" \
    --condition no_skill \
    --max-new-tokens 4096 \
    --temperature 0.7 \
    --top-p 0.9 \
    --seed "$seed"
done
echo "[$(date -u +%FT%TZ)] rollouts complete"

complete=0
for pass in $(seq 1 12); do
  echo "[$(date -u +%FT%TZ)] judge pass=$pass model=$JUDGE_MODEL"
  if ! uv run python code/rlvr/score_rollouts.py \
    --input "$RUN_DIR/rollouts.jsonl" \
    --output "$RUN_DIR/scores.jsonl" \
    --tasks "$RUN_DIR/prompts.jsonl" \
    --judge-model "$JUDGE_MODEL" \
    --judge-protocol paper_pass_fail \
    --max-tokens 2048 \
    --max-tokens-cap 8192 \
    --concurrency 2; then
    echo "[$(date -u +%FT%TZ)] judge transport/preflight failure; retrying in 30s" >&2
    sleep 30
    continue
  fi

  if uv run python code/rlvr/summarize_reward_feasibility.py \
    --scores "$RUN_DIR/scores.jsonl" \
    --rollouts "$RUN_DIR/rollouts.jsonl" \
    --tasks "$RUN_DIR/prompts.jsonl" \
    --output-dir "$RUN_DIR/summary" \
    --judge-model "$JUDGE_MODEL" \
    --judge-protocol paper_pass_fail \
    --seeds "${SEEDS[@]}" \
    --require-complete; then
    complete=1
    break
  fi
  echo "[$(date -u +%FT%TZ)] incomplete after judge pass=$pass; retrying in 30s"
  sleep 30
done

uv run python code/rlvr/summarize_reward_feasibility.py \
  --scores "$RUN_DIR/scores.jsonl" \
  --rollouts "$RUN_DIR/rollouts.jsonl" \
  --tasks "$RUN_DIR/prompts.jsonl" \
  --output-dir "$RUN_DIR/summary" \
  --judge-model "$JUDGE_MODEL" \
  --judge-protocol paper_pass_fail \
  --seeds "${SEEDS[@]}"

if [[ "$complete" -ne 1 ]]; then
  echo "[$(date -u +%FT%TZ)] finished incomplete" >&2
  exit 1
fi
echo "[$(date -u +%FT%TZ)] complete"
