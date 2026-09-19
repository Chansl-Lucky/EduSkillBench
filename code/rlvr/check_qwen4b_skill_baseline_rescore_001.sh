#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if tmux has-session -t eduskill-qwen4b-skill-rescore 2>/dev/null; then
  echo "status=running"
else
  echo "status=not_running"
fi
uv run python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/qwen4b_skill_baseline_rescore_001/scores.jsonl')
rows=[json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []
valid={}
errors=0
for r in rows:
    key=(r.get('task_id'),r.get('condition'),int(r.get('seed',0)))
    if r.get('status')=='ok': valid[key]=r
    else: errors+=1
print(f"valid_unique={len(valid)}/84 raw_records={len(rows)} error_records={errors}")
if valid:
    for condition in ('no_skill','skill_text'):
        vals=[float(r['rubric_reward']) for r in valid.values() if r.get('condition')==condition]
        if vals:
            print(f"{condition}: n={len(vals)} mean={sum(vals)/len(vals):.4f} full_rate={sum(abs(x-1)<1e-9 for x in vals)/len(vals):.4f}")
summary=Path('artifacts/qwen4b_skill_baseline_rescore_001/summary/overall.json')
if summary.exists(): print(summary.read_text())
PY
echo "recent_log:"
tail -12 logs/qwen4b_skill_baseline_rescore_001.log 2>/dev/null || true
