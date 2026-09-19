#!/usr/bin/env python3
"""Aggregate the 84 BenchFlow cells and compute a paired bootstrap CI."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ERROR_FIELDS = (
    "error",
    "verifier_error",
    "export_error",
    "transport_error_info",
    "api_error_info",
    "idle_timeout_info",
    "agent_timeout_info",
    "verifier_timeout_info",
)


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    parts = path.parts
    try:
        skill = parts[parts.index("skill-eval") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot infer skill from {path}") from exc
    mode = "with_skill" if "with-skill" in parts else "no_skill"
    match = re.search(r"__(\d+)__", path.parent.name)
    if not match:
        raise ValueError(f"Cannot infer case number from {path}")
    reward = payload.get("rewards", {}).get("reward")
    clean = reward is not None and not any(payload.get(key) for key in ERROR_FIELDS)
    return {
        "skill": skill,
        "case": match.group(1),
        "mode": mode,
        "reward": reward,
        "clean": clean,
        "finished_at": payload.get("finished_at", ""),
        "n_tool_calls": payload.get("n_tool_calls", 0),
        "n_skill_invocations": payload.get("n_skill_invocations", 0),
        "total_tokens": payload.get("agent_result", {}).get("total_tokens", 0),
        "source": str(path),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=Path("jobs/paper-aligned-qwen3-4b-001"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/paper_aligned_qwen3_4b_001/summary")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in args.jobs.rglob("result.json"):
        row = load_result(path)
        grouped[(row["skill"], row["case"], row["mode"])].append(row)
    chosen = {
        key: sorted(rows, key=lambda row: (row["clean"], row["finished_at"]), reverse=True)[0]
        for key, rows in grouped.items()
    }
    rows = sorted(chosen.values(), key=lambda row: (row["skill"], row["case"], row["mode"]))
    bad = [row for row in rows if not row["clean"]]
    if len(rows) != 84 or bad:
        details = "\n".join(f"{r['skill']} {r['case']} {r['mode']} {r['source']}" for r in bad)
        raise SystemExit(f"Expected 84 clean unique cells; found {len(rows)}, bad={len(bad)}\n{details}")

    pairs: list[tuple[str, str, float, float]] = []
    by_key = {(row["skill"], row["case"], row["mode"]): row for row in rows}
    for skill, case, _ in sorted({(r["skill"], r["case"], "") for r in rows}):
        no_skill = float(by_key[(skill, case, "no_skill")]["reward"])
        with_skill = float(by_key[(skill, case, "with_skill")]["reward"])
        pairs.append((skill, case, no_skill, with_skill))

    lifts = [with_skill - no_skill for _, _, no_skill, with_skill in pairs]
    rng = random.Random(args.seed)
    bootstrap = []
    for _ in range(args.bootstrap_samples):
        sample = [lifts[rng.randrange(len(lifts))] for _ in lifts]
        bootstrap.append(sum(sample) / len(sample))

    by_skill: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for skill, _, no_skill, with_skill in pairs:
        by_skill[skill].append((no_skill, with_skill))
    skill_rows = []
    for skill, values in sorted(by_skill.items()):
        baseline = sum(v[0] for v in values) / len(values)
        augmented = sum(v[1] for v in values) / len(values)
        skill_rows.append(
            {
                "skill": skill,
                "no_skill": baseline,
                "with_skill": augmented,
                "lift": augmented - baseline,
            }
        )

    no_skill_mean = sum(pair[2] for pair in pairs) / len(pairs)
    with_skill_mean = sum(pair[3] for pair in pairs) / len(pairs)
    summary = {
        "experiment_id": "paper_aligned_qwen3_4b_001",
        "policy_model": "qwen3-4b-instruct-2507-cdbee75",
        "primary_judge_model": "qwen3-4b-instruct-2507-cdbee75",
        "runs": len(rows),
        "paired_tasks": len(pairs),
        "no_skill_mean": no_skill_mean,
        "with_skill_mean": with_skill_mean,
        "paired_lift": with_skill_mean - no_skill_mean,
        "paired_lift_bootstrap_95_ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "positive_lift_skills": sum(row["lift"] > 0 for row in skill_rows),
        "zero_lift_skills": sum(row["lift"] == 0 for row in skill_rows),
        "negative_lift_skills": sum(row["lift"] < 0 for row in skill_rows),
        "with_skill_invocation_rate": sum(
            row["n_skill_invocations"] > 0 for row in rows if row["mode"] == "with_skill"
        )
        / len(pairs),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overall.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "skills.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("skill", "no_skill", "with_skill", "lift"))
        writer.writeheader()
        writer.writerows(skill_rows)
    with (args.output / "runs.csv").open("w", newline="") as handle:
        fields = tuple(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
