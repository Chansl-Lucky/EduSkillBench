# Qwen3-4B Skill-Conditioned GRPO Extension

This extension tests whether rubric-guided reinforcement learning can make a trainable small model
use the 14 released single-turn educational Skills more effectively. It is separate from the
original five-model leaderboard: the original paper evaluates inference-time Skill augmentation,
while this experiment additionally updates Qwen3-4B weights with GRPO.

## Research question

> Given the same matched `SKILL.md` at inference time, does a Qwen3-4B policy trained with
> Skill-conditioned GRPO outperform the untrained Qwen3-4B policy?

The three reported conditions are:

1. **Base / No-Skill** — local Qwen3-4B receives only the educational task.
2. **Base / With-Skill** — the same checkpoint receives the full matched `SKILL.md` in the system
   context.
3. **GRPO / With-Skill** — a LoRA-trained checkpoint receives the same matched Skill.

The primary RL comparison is condition 3 minus condition 2. Condition 1 measures the original
inference-time Skill lift.

## Frozen configuration

| Component | Value |
|---|---|
| Policy | `Qwen/Qwen3-4B-Instruct-2507` |
| Revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Harness | local Transformers; full `SKILL.md` system-context injection |
| Skills | 14 single-turn Skills |
| Train/dev | 70 train prompts and 28 dev prompts, disjoint from the official tasks |
| Training | 70 GRPO optimizer steps; 4 completions per step |
| Adapter | LoRA rank 8 on `q/k/v/o` attention projections |
| Rollout sampling | temperature 0.9, top-p 0.9, 2,048-token cap |
| Training reward | continuous task-specific rubric score mixed with completion validity |
| Judge | `deepseek-v4-flash-260425` through Volcengine Ark |
| Final evaluation | 42 official tasks; temperature 0.7, top-p 0.9, seed 20260914, 4,096-token cap |
| Final protocol | BenchFlow-style per-item PASS/FAIL proportion |
| Uncertainty | paired task bootstrap, 20,000 samples |

The official 42 tasks are evaluation-only. They are not used to generate training prompts, compute
training rewards, or select the checkpoint. Character n-gram screening rejects generated prompts
whose similarity to an official task reaches 0.72.

## Results

All three final conditions are scored by the same Ark DS V4 Flash model and the same PASS/FAIL
protocol. Compact machine-readable tables live under `results/qwen3_4b_skill_grpo/`.

| Condition | Mean reward |
|---|---:|
| Base / No-Skill | 0.6435 |
| Base / With-Skill | 0.9063 |
| GRPO / With-Skill | 0.9388 |

Matched Skill injection improves the base model by **+0.2628** (+26.28 percentage points). The
observed GRPO lift over Base / With-Skill is a further **+0.0325** (+3.25 percentage points). Its
paired bootstrap 95% interval is **[-0.0048, 0.0714]**, which crosses zero. The correct
interpretation of the GRPO increment is a positive point estimate, not a statistically stable
positive effect from this single run.

Across the 42 paired tasks, 9 improve, 29 tie, and 4 decline. At Skill level, 6 improve, 6 tie, and
2 decline. Mean completion length changes only from 2,596 to 2,606 tokens, and both conditions have
one truncated response, so the observed lift is not explained by a large length increase.

Training reward is informative but still sparse: 55/70 groups have non-zero reward range, only
8/70 reach range >= 0.2, and 15/70 are exact ties. This weak within-group signal is a plausible
reason the observed RL lift is modest.

## Installation

Requirements:

- Linux with a CUDA-capable GPU (the recorded run used one RTX 4090 with approximately 48 GB
  visible memory)
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- local Qwen3-4B weights
- an OpenAI-compatible Ark endpoint with the configured Judge model enabled

Create the environment:

```bash
bash code/rlvr/setup_rlvr_env.sh
```

Set credentials without writing them to Git:

```bash
cp .env.example .env
# Edit .env locally, then:
set -a
source .env
set +a
```

No real API key is stored in this repository. `.env` is ignored; `.env.example` contains only
masked placeholders.

## Run the three-stage experiment

```bash
export CUDA_VISIBLE_DEVICES=1
bash code/rlvr/run_skill_grpo_formal_001.sh
```

The resumable pipeline performs the following:

1. Generates Base / No-Skill and Base / With-Skill official rollouts.
2. Generates and validates independent train/dev prompt pools for all 14 Skills.
3. Trains a Skill-conditioned LoRA policy with GRPO.
4. Generates GRPO / With-Skill official rollouts.
5. Scores all three conditions with the frozen Judge and exports paired statistics.

Runtime outputs are written under `artifacts/` and `logs/`, both ignored by Git. Check progress with:

```bash
bash code/rlvr/check_skill_grpo_formal_001.sh
```

To export compact release tables after completion:

```bash
uv run python code/rlvr/export_skill_grpo_release_results.py
```

## Important scope boundary

This is a **local weight-level experiment**, not an exact reproduction of the paper's OpenCode +
Docker trajectory harness. It aligns the official tasks, matched Skills, task-specific rubrics,
sampling settings, and final Judge across the three local conditions. The original paper numbers
remain a separate external reference and must not be mixed into this paired comparison.

The main empirical claim is therefore limited to the frozen automated Judge protocol. This run did
not include human evaluation.
