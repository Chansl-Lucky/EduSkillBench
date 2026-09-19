#!/usr/bin/env python3
"""Summarize group-level reward feasibility for independent prompt rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlvr.prompt_pool import load_prompt_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-protocol", default="paper_pass_fail")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--minimum-range", type=float, default=0.2)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [json.loads(line) for line in file_path.read_text().splitlines() if line]


def score_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        row["task_id"],
        row["condition"],
        int(row["seed"]),
        row["judge_model"],
        row.get("judge_protocol", "rubric_continuous"),
    )


def main() -> int:
    args = parse_args()
    tasks = load_prompt_pool(args.tasks)
    expected = {
        (task.task_id, "no_skill", seed, args.judge_model, args.judge_protocol)
        for task in tasks
        for seed in args.seeds
    }
    chosen: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    raw_scores = read_jsonl(args.scores)
    for row in raw_scores:
        key = score_key(row)
        if key not in expected:
            continue
        if row.get("status") == "ok" or key not in chosen:
            chosen[key] = row
    valid = {key: row for key, row in chosen.items() if row.get("status") == "ok"}
    missing = sorted(expected - valid.keys())

    rollouts = read_jsonl(args.rollouts)
    rollout_by_key = {(row["task_id"], row["condition"], int(row["seed"])): row for row in rollouts}
    group_rows: list[dict[str, Any]] = []
    for task in tasks:
        rows = [
            valid[(task.task_id, "no_skill", seed, args.judge_model, args.judge_protocol)]
            for seed in args.seeds
            if (task.task_id, "no_skill", seed, args.judge_model, args.judge_protocol) in valid
        ]
        values = [float(row["rubric_reward"]) for row in rows]
        rollout_rows = [
            rollout_by_key[(task.task_id, "no_skill", seed)]
            for seed in args.seeds
            if (task.task_id, "no_skill", seed) in rollout_by_key
        ]
        score_range = max(values) - min(values) if values else None
        group_rows.append(
            {
                "task_id": task.task_id,
                "skill_id": task.skill_id,
                "valid_scores": len(values),
                "mean_reward": statistics.fmean(values) if values else None,
                "std_reward": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min_reward": min(values) if values else None,
                "max_reward": max(values) if values else None,
                "reward_range": score_range,
                "unique_scores": len(set(values)),
                "nontrivial_range": bool(
                    score_range is not None and score_range >= args.minimum_range
                ),
                "truncated_rollouts": sum(bool(row.get("truncated")) for row in rollout_rows),
                "mean_completion_tokens": (
                    statistics.fmean(float(row["completion_tokens"]) for row in rollout_rows)
                    if rollout_rows
                    else None
                ),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        by_skill[row["skill_id"]].append(row)
    skill_rows: list[dict[str, Any]] = []
    for skill_id, rows in sorted(by_skill.items()):
        complete = [row for row in rows if row["valid_scores"] == len(args.seeds)]
        skill_rows.append(
            {
                "skill_id": skill_id,
                "groups": len(rows),
                "complete_groups": len(complete),
                "mean_reward": (
                    statistics.fmean(float(row["mean_reward"]) for row in complete)
                    if complete
                    else None
                ),
                "nontrivial_group_fraction": (
                    statistics.fmean(float(row["nontrivial_range"]) for row in complete)
                    if complete
                    else None
                ),
            }
        )
    with (output_dir / "skills.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(skill_rows[0]))
        writer.writeheader()
        writer.writerows(skill_rows)

    complete_groups = [row for row in group_rows if row["valid_scores"] == len(args.seeds)]
    json_compliance = len(valid) / len(expected) if expected else 0.0
    nontrivial_fraction = (
        statistics.fmean(float(row["nontrivial_range"]) for row in complete_groups)
        if complete_groups
        else 0.0
    )
    rollout_expected = len(tasks) * len(args.seeds)
    truncated = sum(bool(row.get("truncated")) for row in rollouts)
    overall = {
        "experiment": "reward_feasibility_001",
        "judge_model": args.judge_model,
        "judge_protocol": args.judge_protocol,
        "tasks": len(tasks),
        "seeds_per_task": len(args.seeds),
        "expected_scores": len(expected),
        "valid_scores": len(valid),
        "missing_scores": len(missing),
        "json_protocol_success_rate": json_compliance,
        "complete_groups": len(complete_groups),
        "nontrivial_group_fraction": nontrivial_fraction,
        "all_tie_group_fraction": (
            statistics.fmean(float(row["unique_scores"] == 1) for row in complete_groups)
            if complete_groups
            else 1.0
        ),
        "mean_group_std": (
            statistics.fmean(float(row["std_reward"]) for row in complete_groups)
            if complete_groups
            else 0.0
        ),
        "rollouts": len(rollouts),
        "expected_rollouts": rollout_expected,
        "truncated_rollouts": truncated,
        "truncation_rate": truncated / rollout_expected if rollout_expected else 0.0,
        "automatic_gates": {
            "json_protocol_at_least_0_99": json_compliance >= 0.99,
            "nontrivial_groups_at_least_0_70": nontrivial_fraction >= 0.70,
        },
        "pending_gates": [
            "subject_matter_review_of_generated_prompts",
            "judge_repeatability_on_identical_responses",
            "human_pairwise_agreement",
            "length_and_style_bias_audit",
        ],
        "decision": "pending_human_and_repeatability_gates",
        "missing_keys": [list(key) for key in missing],
        "raw_score_records": len(raw_scores),
    }
    (output_dir / "overall.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    incomplete = bool(missing) or len(rollouts) != rollout_expected
    return 1 if args.require_complete and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
