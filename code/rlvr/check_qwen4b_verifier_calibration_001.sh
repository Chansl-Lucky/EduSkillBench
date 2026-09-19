#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN_DIR="artifacts/qwen4b_verifier_calibration_001"
if tmux has-session -t eduskill-qwen4b-verifier 2>/dev/null; then
  echo "status=running"
else
  echo "status=not_running"
fi

uv run python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict

p = Path("artifacts/qwen4b_verifier_calibration_001/scores.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
valid = {}
errors = 0
for row in rows:
    key = (row.get("task_id"), row.get("condition"), int(row.get("seed", 0)))
    if row.get("status") == "ok":
        valid[key] = row
    else:
        errors += 1
rewards = [float(row["reward"]) for row in valid.values()]
print(f"valid_unique={len(valid)}/168 raw_records={len(rows)} error_records={errors}")
if rewards:
    print(f"mean={sum(rewards)/len(rewards):.4f} min={min(rewards):.4f} max={max(rewards):.4f}")
    print(f"full_score_rate={sum(abs(x-1.0)<1e-9 for x in rewards)/len(rewards):.4f}")
summary = Path("artifacts/qwen4b_verifier_calibration_001/summary/overall.json")
if summary.exists():
    print(summary.read_text())
PY

echo "recent_log:"
tail -12 logs/qwen4b_verifier_calibration_001.log 2>/dev/null || true
