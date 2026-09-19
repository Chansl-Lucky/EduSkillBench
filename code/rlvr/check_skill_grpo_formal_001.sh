#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
RUN_DIR="artifacts/skill_grpo_formal_001"

if tmux has-session -t eduskill-skill-grpo-formal 2>/dev/null; then
  echo "status=running tmux=eduskill-skill-grpo-formal"
else
  echo "status=not_running"
fi

train_pools=$(find "$RUN_DIR/train_pools" -type f -name '*.jsonl' -size +0c 2>/dev/null | wc -l)
dev_pools=$(find "$RUN_DIR/dev_pools" -type f -name '*.jsonl' -size +0c 2>/dev/null | wc -l)
echo "generated_train_skill_pools=$train_pools/14"
echo "generated_dev_skill_pools=$dev_pools/14"

if [[ -f "$RUN_DIR/train_prompts.jsonl" ]]; then
  echo "assembled_train_prompts=$(wc -l < "$RUN_DIR/train_prompts.jsonl")/70"
fi
if [[ -f "$RUN_DIR/dev_prompts.jsonl" ]]; then
  echo "assembled_dev_prompts=$(wc -l < "$RUN_DIR/dev_prompts.jsonl")/28"
fi
if [[ -f "$RUN_DIR/reward_trace.jsonl" ]]; then
  rewards=$(wc -l < "$RUN_DIR/reward_trace.jsonl")
  echo "training_reward_records=$rewards/280"
fi

latest_checkpoint=""
if [[ -d "$RUN_DIR/checkpoint" ]]; then
  latest_checkpoint=$(find "$RUN_DIR/checkpoint" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)
fi
if [[ -n "$latest_checkpoint" ]]; then
  echo "latest_checkpoint=$latest_checkpoint"
fi

for name in base_skill_rollouts grpo_skill_rollouts base_skill_scores grpo_skill_scores; do
  path="$RUN_DIR/eval/$name.jsonl"
  if [[ -f "$path" ]]; then
    echo "$name=$(wc -l < "$path")"
  fi
done

if [[ -f "$RUN_DIR/eval/comparison.json" ]]; then
  echo "comparison:"
  cat "$RUN_DIR/eval/comparison.json"
fi

echo "log_tail:"
tail -20 logs/skill_grpo_formal_001.log 2>/dev/null || true
