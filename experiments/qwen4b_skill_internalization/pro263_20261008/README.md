# Qwen3-4B Skill internalization: 263-task Pro evaluation

See [REPORT.md](REPORT.md) for the four-arm result table, [paper_report.json](paper_report.json)
for paired source-cluster intervals and per-Skill results, and [paper_rows.json](paper_rows.json)
for all task-level outcomes. Failed execution/Judge cells remain in the fixed denominator;
they are not removed or replaced by the best repeat. Inspect failure counts before interpreting gains.

## Experiment

- Base: original Qwen3-4B-Instruct-2507, no Skill.
- Base+Skill: original model, matched Skill and accessible resources.
- SFT: LoRA distilled from the accepted 246-task training release, no Skill.
- GRPO: 123 outcome-based optimizer steps starting from that SFT adapter, no Skill.
- Test: only the 263 advisory tasks; 9 Skills and 30 source documents (not all 14 training Skills).
- Generation: existing BenchFlow/Docker/OpenCode 1.18.11, education-single-turn system,
  600-second task timeout. GRPO replaces only the trained adapter/alias and server port.
- Judge: requested deepseek-v4-pro, resolved deepseek-v4-pro-ga-260813; Ark Plan Chat,
  thinking disabled, max output 10000, original streaming deadlines, no explicit temperature/seed.
- Scoring: pinned upstream commit 6270c70e793ad625297dcce8074bc5cd601aa5c1,
  source-native-v1 prompt and validation; parent final answer only.
- Main statistic: highest-native-label fraction per task, then macro mean over 263;
  terminal failures zero. 5000 paired source-cluster bootstrap resamples, seed 20261005.

The upstream formal experiment generated answers via direct Chat. Our Docker/OpenCode
generation is still different. Scoring alignment must not be described as full environment equality.
Old Flash scores used full multi-agent trajectories and are not mixed into this Pro table.
The previous three-arm Pro run retained unresolved Judge validation failures; report them explicitly.

## Run on the experiment server

The launch documentation and original manifests are included. Local model weights,
Docker jobs, API keys, account credentials and raw request logs are deliberately excluded.
Use separate environments: the existing .venv-eval305-handoff for BenchFlow,
.venv-qwen4b-serve for vLLM, and .venv for training/reporting. Do not upgrade their dependencies
mid-comparison. Restore pinned upstream files under the documented snapshot paths and
provide the Qwen model + completed adapter locally.

```bash
export PYTHONPATH="$PWD/code:$PWD/code/evaluation"
export LLM_API_KEY='YOUR_ARK_PLAN_KEY'
.venv-eval305-handoff/bin/python code/evaluation/eval_grpo263_pro_20261008.py
```

This evaluation only generates GRPO answers; no Base/SFT reroll or training occurs.
Generation concurrency is 8, Judge capacity 64; CPU-only scoring overlaps generation.
The model server is terminated on completion/error. Existing successful cells are cached.
The publication helper uses an isolated clone of the user's repository and a secrets scan.
These scripts reflect a server handoff, not a self-contained one-command install without models/assets.
