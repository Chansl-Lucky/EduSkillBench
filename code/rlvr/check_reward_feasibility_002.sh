#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

RUN_DIR="artifacts/reward_feasibility_002"
PID_FILE="logs/reward_feasibility_002.pid"
LOG_FILE="logs/reward_feasibility_002.log"
SUMMARY_FILE="$RUN_DIR/summary/overall.json"
SCORES_FILE="$RUN_DIR/dense_scores.jsonl"

score_records=0
valid_scores=0
if [[ -f "$SCORES_FILE" ]]; then
  score_records=$(wc -l < "$SCORES_FILE")
  valid_scores=$(uv run python - "$SCORES_FILE" <<'PY'
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
if [[ -f "$SUMMARY_FILE" ]] && jq -e '.valid_scores == 168 and .missing_scores == 0' \
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
echo "frozen_prompts=42/42 frozen_rollouts=168/168"
echo "dense_valid_scores=$valid_scores/168 raw_dense_records=$score_records"
remaining=$((168 - valid_scores)); ((remaining < 0)) && remaining=0
echo "rough_eta_minutes=$(( (remaining * 25 + 59) / 60 )) safe_window_hours=2"

if [[ -f "$SUMMARY_FILE" ]]; then
  echo "dense_summary:"
  uv run python -m json.tool "$SUMMARY_FILE"
fi
if [[ -f "$RUN_DIR/summary/protocol_comparison.json" ]]; then
  echo "protocol_comparison:"
  uv run python -m json.tool "$RUN_DIR/summary/protocol_comparison.json"
fi
if [[ -f "$LOG_FILE" ]]; then
  echo "recent_log:"
  tail -20 "$LOG_FILE"
fi
