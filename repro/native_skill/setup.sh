#!/usr/bin/env bash
set -euo pipefail
TASK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ENV_PATH="${NATIVE_SKILL_ENV:-$TASK_REPO_ROOT/.venv-native-skill}"
cd "$TASK_REPO_ROOT"
python3.12 -m venv "$TASK_ENV_PATH"
"$TASK_ENV_PATH/bin/python" -m pip install -r repro/native_skill/environment/benchflow-observed-requirements.txt
"$TASK_ENV_PATH/bin/python" -m repro.native_skill.install_overlay
docker build -t eduskillbench-opencode-runtime:1.18.11 -f repro/native_skill/docker/Dockerfile .
docker build -t eduskillbench-opencode-runtime:1.18.11-rg1 -f repro/native_skill/docker/Dockerfile.rg .
"$TASK_ENV_PATH/bin/python" -m repro.native_skill.verify

