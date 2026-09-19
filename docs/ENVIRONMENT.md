# Reproducible experiment environment

The RL environment is isolated in the repository-local `.venv`. It targets
Python 3.12 to match the released EduSkillBench setup.

## Create or refresh the RL environment

```bash
uv venv --clear --python 3.12 .venv
uv sync --extra train --extra dev
```

Activate it when working interactively:

```bash
source .venv/bin/activate
```

Prefer `uv run ...` in scripts and experiment manifests so commands do not
depend on the caller having activated the environment.

## Original evaluation environment

The released benchmark used BenchFlow 0.6.7 and Docker. Install the pinned
Python component only when reproducing the original evaluation:

```bash
# Keep the RL stack installed as well:
uv sync --extra train --extra dev --extra eval
uv run bench --version
```

For an evaluation-only checkout, `uv sync --extra eval` is sufficient, but it
will intentionally remove optional training packages from the environment.

Docker is a host-level dependency and is intentionally not installed or
managed by this Python environment. The RL feasibility pipeline does not
require Docker; the final official 42-task evaluation does.

## CUDA policy

The current host driver reports CUDA 12.4. The project therefore resolves
PyTorch 2.6.0 from the official `cu124` wheel index. Do not install vLLM into
this environment until its Torch/CUDA compatibility has been validated.

The initial pilot uses TRL generation on one GPU. A separate rollout-serving
environment may be added later for vLLM so it cannot silently replace the
trainer's Torch build.

## Secrets

Set TokenPlan credentials only in the shell:

```bash
export TOKENPLAN_API_KEY="..."
export TOKENPLAN_BASE_URL="https://discovery-api.intern-ai.org.cn/v1"
```

`.env` files are ignored by Git, but shell or secret-manager injection is
preferred. Experiment manifests must record only the base URL and model ID,
never the credential.
