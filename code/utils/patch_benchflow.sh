#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---apply}"
case "$MODE" in
  --apply|--check) ;;
  *) echo "usage: $0 [--apply|--check]" >&2; exit 2 ;;
esac

# Locate the package from this repository's uv environment.  Do not assume a
# globally installed `bench` tool and do not use the system Python.
ROOT=$(uv run python - <<'PY'
from pathlib import Path
import benchflow
print(Path(benchflow.__file__).resolve().parent)
PY
)
CORE="$ROOT/skill_eval/_core.py"
JUDGE="$ROOT/templates/judge.py.tmpl"

uv run python - "$MODE" "$CORE" "$JUDGE" <<'PY'
from pathlib import Path
import shutil
import sys

mode, core_name, judge_name = sys.argv[1:]
core = Path(core_name)
judge = Path(judge_name)

def is_patched() -> bool:
    c = core.read_text()
    j = judge.read_text()
    return (
        '"OPENAI_BASE_URL",' in c
        and 'model=model,' in j
        and 'and not os.environ.get("OPENAI_BASE_URL")' in j
    )

if mode == "--check":
    if not is_patched():
        raise SystemExit(
            "BenchFlow local-judge patch is NOT applied. Run code/utils/patch_benchflow.sh --apply"
        )
    print(f"BenchFlow local-judge patch OK: {core.parent.parent}")
    raise SystemExit(0)

for path in (core, judge):
    backup = path.with_suffix(path.suffix + ".eduskillbench.orig")
    if not backup.exists():
        shutil.copy2(path, backup)

c = core.read_text()
anchor = '    "OPENAI_API_KEY",\n'
if anchor not in c:
    raise SystemExit(f"Unexpected BenchFlow core layout: {core}")
for entry in ('    "OPENAI_BASE_URL",\n', '    "NO_PROXY",\n', '    "no_proxy",\n'):
    if entry not in c:
        c = c.replace(anchor, anchor + entry, 1)
core.write_text(c)

j = judge.read_text()
old_model = 'model=model if is_openai_model else "gpt-4o-mini",'
if old_model in j:
    j = j.replace(old_model, 'model=model,', 1)
elif 'model=model,' not in j:
    raise SystemExit(f"Unexpected BenchFlow judge model expression: {judge}")

# A custom OpenAI endpoint is an explicit routing decision.  Avoid first trying
# the Anthropic SDK merely because the local alias does not start with `gpt-`.
old_anthropic = 'if not is_openai_model and not is_gemini_model:'
new_anthropic = (
    'if (not is_openai_model and not is_gemini_model '
    'and not os.environ.get("OPENAI_BASE_URL")):'
)
if old_anthropic in j:
    j = j.replace(old_anthropic, new_anthropic, 1)
elif new_anthropic not in j:
    raise SystemExit(f"Unexpected BenchFlow judge provider branch: {judge}")
judge.write_text(j)

if not is_patched():
    raise SystemExit("Patch validation failed")
print(f"BenchFlow local-judge patch applied: {core.parent.parent}")
PY
