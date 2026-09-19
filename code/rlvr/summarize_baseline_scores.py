#!/usr/bin/env python3
"""Aggregate paired RLVR baseline scores using the paper's macro-average convention."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlvr.data import load_official_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="data/single_turn_tasks.csv")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    args = parse_args()
    tasks = load_official_tasks(args.tasks)
    task_by_id = {task.task_id: task for task in tasks}
    expected = {
        (task.task_id, condition, args.seed, args.judge_model)
        for task in tasks
        for condition in ("no_skill", "skill_text")
    }

    # Later successful retries replace earlier protocol errors for the same cell.
    chosen: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for record in read_jsonl(Path(args.input)):
        key = (
            record["task_id"],
            record["condition"],
            int(record["seed"]),
            record["judge_model"],
        )
        if key not in expected:
            continue
        if record.get("status") == "ok" or key not in chosen:
            chosen[key] = record

    valid = {key: record for key, record in chosen.items() if record.get("status") == "ok"}
    missing = sorted(expected - valid.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for key, record in sorted(valid.items()):
        task = task_by_id[key[0]]
        run_rows.append(
            {
                "task_id": task.task_id,
                "skill_id": task.skill_id,
                "difficulty": task.difficulty,
                "condition": record["condition"],
                "seed": record["seed"],
                "judge_model": record["judge_model"],
                "rubric_reward": record["rubric_reward"],
                "mixed_reward": record["reward"],
                "hard_reward": record["hard_reward"],
                "failure_penalty": record["failure_penalty"],
                "truncated": record["truncated"],
                "completion_tokens": record["completion_tokens"],
                "judge_max_tokens": record["judge"]["max_tokens_used"],
            }
        )

    fields = list(run_rows[0]) if run_rows else []
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(run_rows)

    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_skill[row["skill_id"]].append(row)
    skill_rows: list[dict[str, Any]] = []
    for skill_id in sorted({task.skill_id for task in tasks}):
        rows = by_skill[skill_id]
        no_skill = [row for row in rows if row["condition"] == "no_skill"]
        with_skill = [row for row in rows if row["condition"] == "skill_text"]

        def mean(items: list[dict[str, Any]], field: str) -> float | None:
            return sum(float(item[field]) for item in items) / len(items) if items else None

        no_rubric = mean(no_skill, "rubric_reward")
        with_rubric = mean(with_skill, "rubric_reward")
        skill_rows.append(
            {
                "skill_id": skill_id,
                "valid_no_skill": len(no_skill),
                "valid_with_skill": len(with_skill),
                "no_skill_rubric": no_rubric,
                "with_skill_rubric": with_rubric,
                "rubric_lift": (
                    with_rubric - no_rubric
                    if no_rubric is not None and with_rubric is not None
                    else None
                ),
                "no_skill_mixed": mean(no_skill, "mixed_reward"),
                "with_skill_mixed": mean(with_skill, "mixed_reward"),
                "no_skill_truncated": sum(bool(row["truncated"]) for row in no_skill),
                "with_skill_truncated": sum(bool(row["truncated"]) for row in with_skill),
            }
        )

    with (output_dir / "skills.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(skill_rows[0]))
        writer.writeheader()
        writer.writerows(skill_rows)

    no_skill = [row for row in run_rows if row["condition"] == "no_skill"]
    with_skill = [row for row in run_rows if row["condition"] == "skill_text"]

    def overall_mean(items: list[dict[str, Any]], field: str) -> float | None:
        return sum(float(item[field]) for item in items) / len(items) if items else None

    no_rubric = overall_mean(no_skill, "rubric_reward")
    with_rubric = overall_mean(with_skill, "rubric_reward")
    lift = with_rubric - no_rubric if no_rubric is not None and with_rubric is not None else None
    complete_skills = [
        row for row in skill_rows if row["valid_no_skill"] == 3 and row["valid_with_skill"] == 3
    ]
    overall = {
        "protocol": "official_42_paired_single_run_weight_level_baseline",
        "primary_metric": "rubric_reward",
        "judge_model": args.judge_model,
        "seed": args.seed,
        "expected_runs": len(expected),
        "valid_runs": len(valid),
        "missing_runs": len(missing),
        "missing_keys": [list(key) for key in missing],
        "no_skill_rubric": no_rubric,
        "with_skill_rubric": with_rubric,
        "rubric_lift": lift,
        "normalized_gain": (
            lift / (1 - no_rubric)
            if lift is not None and no_rubric is not None and no_rubric < 1
            else None
        ),
        "no_skill_mixed": overall_mean(no_skill, "mixed_reward"),
        "with_skill_mixed": overall_mean(with_skill, "mixed_reward"),
        "no_skill_truncated": sum(bool(row["truncated"]) for row in no_skill),
        "with_skill_truncated": sum(bool(row["truncated"]) for row in with_skill),
        "positive_lift_skills": sum(float(row["rubric_lift"]) > 0 for row in complete_skills),
        "zero_lift_skills": sum(float(row["rubric_lift"]) == 0 for row in complete_skills),
        "negative_lift_skills": sum(float(row["rubric_lift"]) < 0 for row in complete_skills),
        "complete_skills": len(complete_skills),
    }
    (output_dir / "overall.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 1 if args.require_complete and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
