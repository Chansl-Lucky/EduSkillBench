#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${ARK_API_KEY:-}" ]]; then
  echo "ARK_API_KEY is required" >&2
  exit 2
fi

export TOKENPLAN_API_KEY="$ARK_API_KEY"
export TOKENPLAN_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export PYTHONPATH=code
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ID="skill_grpo_formal_001"
RUN_DIR="artifacts/$RUN_ID"
TRAIN_POOL_DIR="$RUN_DIR/train_pools"
DEV_POOL_DIR="$RUN_DIR/dev_pools"
MODEL="${QWEN3_4B_MODEL_PATH:-/home/gpuuser/csl/models/Qwen3-4B-Instruct-2507}"
JUDGE_MODEL="${ARK_JUDGE_MODEL:-deepseek-v4-flash-260425}"
mkdir -p "$TRAIN_POOL_DIR" "$DEV_POOL_DIR" "$RUN_DIR/eval"

mapfile -t SKILLS < <(find skills/single_turn -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
[[ "${#SKILLS[@]}" -eq 14 ]] || { echo "expected 14 Skill directories" >&2; exit 1; }

echo "[$(date -u +%FT%TZ)] stage=base_rollouts"
.venv/bin/python code/rlvr/run_local_baseline.py \
  --model "$MODEL" \
  --tasks data/single_turn_tasks.csv \
  --condition both \
  --output "$RUN_DIR/eval/base_rollouts.jsonl" \
  --max-new-tokens 4096 \
  --temperature 0.7 --top-p 0.9 --seed 20260914

generate_pool() {
  local skill_id="$1"
  local split="$2"
  local count="$3"
  local output="$4"
  if [[ -s "$output" ]]; then
    echo "[$(date -u +%FT%TZ)] reuse $output"
    return
  fi
  for attempt in 1 2 3; do
    echo "[$(date -u +%FT%TZ)] generate split=$split skill=$skill_id attempt=$attempt"
    if .venv/bin/python code/rlvr/generate_prompt_pool.py \
      --skill-id "$skill_id" --split "$split" --count "$count" \
      --model "$JUDGE_MODEL" --output "$output" --max-tokens 8192; then
      return
    fi
    sleep 10
  done
  echo "failed generating $split pool for $skill_id" >&2
  exit 1
}

echo "[$(date -u +%FT%TZ)] stage=generate_train_dev"
for skill_id in "${SKILLS[@]}"; do
  generate_pool "$skill_id" train 5 "$TRAIN_POOL_DIR/$skill_id.jsonl"
  generate_pool "$skill_id" dev 2 "$DEV_POOL_DIR/$skill_id.jsonl"
done

.venv/bin/python code/rlvr/assemble_prompt_pools.py \
  --input-dir "$TRAIN_POOL_DIR" --output "$RUN_DIR/train_prompts.jsonl" \
  --report "$RUN_DIR/train_prompt_audit.json" --expected-files 14 --expected-records 70
.venv/bin/python code/rlvr/assemble_prompt_pools.py \
  --input-dir "$DEV_POOL_DIR" --output "$RUN_DIR/dev_prompts.jsonl" \
  --report "$RUN_DIR/dev_prompt_audit.json" --expected-files 14 --expected-records 28

echo "[$(date -u +%FT%TZ)] stage=train_grpo"
.venv/bin/python code/rlvr/train_skill_grpo.py \
  --prompts "$RUN_DIR/train_prompts.jsonl" \
  --model "$MODEL" \
  --output-dir "$RUN_DIR" \
  --judge-model "$JUDGE_MODEL" \
  --base-url "$TOKENPLAN_BASE_URL" \
  --max-steps 70 \
  --num-generations 4 \
  --max-completion-length 2048 \
  --save-steps 10 \
  --resume

echo "[$(date -u +%FT%TZ)] stage=official_eval_generation"
.venv/bin/python code/rlvr/prepare_skill_grpo_eval.py \
  --input "$RUN_DIR/eval/base_rollouts.jsonl" \
  --output "$RUN_DIR/eval/base_skill_rollouts.jsonl" \
  --condition skill_text
.venv/bin/python code/rlvr/prepare_skill_grpo_eval.py \
  --input "$RUN_DIR/eval/base_rollouts.jsonl" \
  --output "$RUN_DIR/eval/base_no_skill_rollouts.jsonl" \
  --condition no_skill
.venv/bin/python code/rlvr/run_local_baseline.py \
  --model "$MODEL" \
  --adapter "$RUN_DIR/final_adapter" \
  --tasks data/single_turn_tasks.csv \
  --condition skill_text \
  --output "$RUN_DIR/eval/grpo_skill_rollouts.jsonl" \
  --max-new-tokens 4096 \
  --temperature 0.7 --top-p 0.9 --seed 20260914

echo "[$(date -u +%FT%TZ)] stage=official_eval_scoring"
.venv/bin/python code/rlvr/score_rollouts.py \
  --input "$RUN_DIR/eval/base_no_skill_rollouts.jsonl" \
  --output "$RUN_DIR/eval/base_no_skill_scores.jsonl" \
  --tasks data/single_turn_tasks.csv \
  --judge-model "$JUDGE_MODEL" --judge-protocol paper_pass_fail \
  --max-tokens 2048 --max-tokens-cap 8192 --concurrency 2
.venv/bin/python code/rlvr/score_rollouts.py \
  --input "$RUN_DIR/eval/base_skill_rollouts.jsonl" \
  --output "$RUN_DIR/eval/base_skill_scores.jsonl" \
  --tasks data/single_turn_tasks.csv \
  --judge-model "$JUDGE_MODEL" --judge-protocol paper_pass_fail \
  --max-tokens 2048 --max-tokens-cap 8192 --concurrency 2
.venv/bin/python code/rlvr/score_rollouts.py \
  --input "$RUN_DIR/eval/grpo_skill_rollouts.jsonl" \
  --output "$RUN_DIR/eval/grpo_skill_scores.jsonl" \
  --tasks data/single_turn_tasks.csv \
  --judge-model "$JUDGE_MODEL" --judge-protocol paper_pass_fail \
  --max-tokens 2048 --max-tokens-cap 8192 --concurrency 2
.venv/bin/python code/rlvr/compare_skill_grpo_eval.py \
  --no-skill "$RUN_DIR/eval/base_no_skill_scores.jsonl" \
  --base "$RUN_DIR/eval/base_skill_scores.jsonl" \
  --grpo "$RUN_DIR/eval/grpo_skill_scores.jsonl" \
  --output "$RUN_DIR/eval/comparison.json"

echo "[$(date -u +%FT%TZ)] complete run_id=$RUN_ID"
