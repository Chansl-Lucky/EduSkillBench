#!/usr/bin/env python3
"""Export compact, versionable result tables from the ignored GRPO artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def last_ok(path: Path) -> dict[str, dict]:
    output = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            output[record["task_id"]] = record
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/skill_grpo_formal_001")
    parser.add_argument("--output-dir", default="results/qwen3_4b_skill_grpo")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path("data/single_turn_tasks.csv").open(newline="", encoding="utf-8") as handle:
        task_skill = {row["task_id"]: row["skill_id"] for row in csv.DictReader(handle)}
    no_skill = last_ok(run_dir / "eval/base_no_skill_scores.jsonl")
    base_skill = last_ok(run_dir / "eval/base_skill_scores.jsonl")
    grpo_skill = last_ok(run_dir / "eval/grpo_skill_scores.jsonl")
    shared = sorted(set(no_skill) & set(base_skill) & set(grpo_skill))
    if len(shared) != 42:
        raise SystemExit(f"expected 42 complete three-way task cells, found {len(shared)}")

    task_rows = []
    for task_id in shared:
        ns = float(no_skill[task_id]["rubric_reward"])
        bs = float(base_skill[task_id]["rubric_reward"])
        gs = float(grpo_skill[task_id]["rubric_reward"])
        task_rows.append(
            {
                "task_id": task_id,
                "skill_id": task_skill[task_id],
                "base_no_skill": ns,
                "base_with_skill": bs,
                "grpo_with_skill": gs,
                "skill_lift": bs - ns,
                "grpo_lift_over_base_skill": gs - bs,
            }
        )

    by_skill: dict[str, list[dict]] = defaultdict(list)
    for row in task_rows:
        by_skill[row["skill_id"]].append(row)
    skill_rows = []
    for skill_id, rows in sorted(by_skill.items()):
        skill_rows.append(
            {
                "skill_id": skill_id,
                "tasks": len(rows),
                "base_no_skill": sum(r["base_no_skill"] for r in rows) / len(rows),
                "base_with_skill": sum(r["base_with_skill"] for r in rows) / len(rows),
                "grpo_with_skill": sum(r["grpo_with_skill"] for r in rows) / len(rows),
                "skill_lift": sum(r["skill_lift"] for r in rows) / len(rows),
                "grpo_lift_over_base_skill": sum(r["grpo_lift_over_base_skill"] for r in rows)
                / len(rows),
            }
        )

    ns = np.array([row["base_no_skill"] for row in task_rows])
    bs = np.array([row["base_with_skill"] for row in task_rows])
    gs = np.array([row["grpo_with_skill"] for row in task_rows])
    delta = gs - bs
    rng = np.random.default_rng(20260918)
    boot = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
    trace = [
        json.loads(line)
        for line in (run_dir / "reward_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ranges = []
    for offset in range(0, len(trace), 4):
        rewards = [float(item["reward"]) for item in trace[offset : offset + 4]]
        ranges.append(max(rewards) - min(rewards))
    summary = {
        "experiment": "Qwen3-4B Skill-conditioned GRPO",
        "policy_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "harness": "Transformers with full SKILL.md system-context injection",
        "judge": "deepseek-v4-flash-260425 via Volcengine Ark",
        "final_protocol": "BenchFlow-style rubric item PASS/FAIL proportion",
        "official_tasks": 42,
        "skills": 14,
        "train_prompts": 70,
        "grpo_steps": 70,
        "generations_per_step": 4,
        "base_no_skill_mean": float(ns.mean()),
        "base_with_skill_mean": float(bs.mean()),
        "skill_lift": float((bs - ns).mean()),
        "grpo_with_skill_mean": float(gs.mean()),
        "grpo_lift_over_base_skill": float(delta.mean()),
        "grpo_lift_bootstrap_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "task_positive_tie_negative": [
            int((delta > 0).sum()),
            int((delta == 0).sum()),
            int((delta < 0).sum()),
        ],
        "skill_positive_tie_negative": [
            sum(row["grpo_lift_over_base_skill"] > 0 for row in skill_rows),
            sum(row["grpo_lift_over_base_skill"] == 0 for row in skill_rows),
            sum(row["grpo_lift_over_base_skill"] < 0 for row in skill_rows),
        ],
        "training_groups": len(ranges),
        "training_groups_nonzero_range": sum(value > 0 for value in ranges),
        "training_groups_range_at_least_0_2": sum(value >= 0.2 for value in ranges),
        "mean_training_group_range": float(np.mean(ranges)),
        "interpretation": (
            "Positive point estimate; the paired bootstrap interval crosses zero, so this run "
            "does not establish a statistically stable positive GRPO effect."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "skills.csv", skill_rows)
    write_csv(output_dir / "tasks.csv", task_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
