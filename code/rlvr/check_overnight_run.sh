#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

PID_FILE="logs/overnight_dsflash_20260914.pid"
LOG_FILE="logs/overnight_dsflash_20260914.log"
RUN_DIR="artifacts/overnight_dsflash_20260914"

SUMMARY_FILE="$RUN_DIR/summary/overall.json"
run_complete=0
if [[ -f "$SUMMARY_FILE" ]] && jq -e '.missing_runs == 0 and .valid_runs == .expected_runs' \
  "$SUMMARY_FILE" >/dev/null 2>&1; then
  run_complete=1
fi

if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "status=running pid=$pid"
  elif [[ "$run_complete" -eq 1 ]]; then
    echo "status=complete last_pid=$pid"
  else
    echo "status=not_running last_pid=$pid"
  fi
elif [[ "$run_complete" -eq 1 ]]; then
  echo "status=complete pid_file_missing"
else
  echo "status=pid_file_missing"
fi

count_lines() {
  if [[ -f "$1" ]]; then
    wc -l < "$1"
  else
    echo 0
  fi
}

echo "rollouts=$(count_lines "$RUN_DIR/rollouts.jsonl")/84"
echo "score_records=$(count_lines "$RUN_DIR/scores.jsonl")"

if [[ -f "$RUN_DIR/scores.jsonl" ]]; then
  uv run python - "$RUN_DIR/scores.jsonl" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line]
by_key = defaultdict(list)
for row in rows:
    key = (
        row["task_id"],
        row["condition"],
        int(row["seed"]),
        row["judge_model"],
        row.get("judge_protocol", "rubric_continuous"),
    )
    by_key[key].append(row)
valid = sum(any(row.get("status") == "ok" for row in values) for values in by_key.values())
errors = Counter(
    row.get("error_type", "unknown") for row in rows if row.get("status") != "ok"
)
print(f"valid_unique_scores={valid}/84")
print(f"missing_unique_scores={84-valid}")
print(f"duplicate_retry_records={len(rows)-len(by_key)}")
print(f"error_records={dict(errors)}")
PY
fi

if [[ -f "$SUMMARY_FILE" ]]; then
  echo "overall_summary:"
  uv run python -m json.tool "$SUMMARY_FILE"
fi

echo "time_budget_guidance=about_2h_active_typical; reserve_3h_with_provider_retries"

if [[ -f "$LOG_FILE" ]]; then
  echo "recent_log:"
  tail -20 "$LOG_FILE"
fi
