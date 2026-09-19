#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

RUN_DIR="artifacts/reward_feasibility_001"
PID_FILE="logs/reward_feasibility_001.pid"
LOG_FILE="logs/reward_feasibility_001.log"
SUMMARY_FILE="$RUN_DIR/summary/overall.json"

count_lines() {
  [[ -f "$1" ]] && wc -l < "$1" || echo 0
}

pool_records=0
for path in "$RUN_DIR"/pools/*.jsonl; do
  [[ -f "$path" ]] || continue
  pool_records=$((pool_records + $(wc -l < "$path")))
done
rollouts=$(count_lines "$RUN_DIR/rollouts.jsonl")

valid_scores=0
score_records=$(count_lines "$RUN_DIR/scores.jsonl")
if [[ -f "$RUN_DIR/scores.jsonl" ]]; then
  valid_scores=$(uv run python - "$RUN_DIR/scores.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x]
keys={(r['task_id'],r['condition'],int(r['seed']),r['judge_model'],r.get('judge_protocol'))
      for r in rows if r.get('status')=='ok'}
print(len(keys))
PY
)
fi

complete=0
if [[ -f "$SUMMARY_FILE" ]] && jq -e '.valid_scores == .expected_scores and .rollouts == .expected_rollouts' \
  "$SUMMARY_FILE" >/dev/null 2>&1; then
  complete=1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "status=running pid=$(cat "$PID_FILE")"
elif [[ "$complete" -eq 1 ]]; then
  echo "status=complete"
else
  echo "status=not_running"
fi

echo "candidate_prompts=$pool_records/42"
echo "rollouts=$rollouts/168"
echo "valid_scores=$valid_scores/168 raw_score_records=$score_records"

remaining_prompts=$((42 - pool_records)); ((remaining_prompts < 0)) && remaining_prompts=0
remaining_rollouts=$((168 - rollouts)); ((remaining_rollouts < 0)) && remaining_rollouts=0
remaining_scores=$((168 - valid_scores)); ((remaining_scores < 0)) && remaining_scores=0
eta_seconds=$((remaining_prompts * 30 + remaining_rollouts * 35 + remaining_scores * 12))
echo "rough_eta_minutes=$(( (eta_seconds + 59) / 60 )) safe_total_window_hours=4"

if [[ -f "$SUMMARY_FILE" ]]; then
  echo "overall_summary:"
  uv run python -m json.tool "$SUMMARY_FILE"
fi
if [[ -f "$LOG_FILE" ]]; then
  echo "recent_log:"
  tail -20 "$LOG_FILE"
fi
