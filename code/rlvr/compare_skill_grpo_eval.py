#!/usr/bin/env python3
"""Compare paired Base+Skill and GRPO+Skill Judge scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def last_ok(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            result[record["task_id"]] = record
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--grpo", required=True)
    parser.add_argument("--no-skill", help="Optional Base/No-Skill scores from the same Judge")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    base = last_ok(Path(args.base))
    grpo = last_ok(Path(args.grpo))
    shared = sorted(set(base) & set(grpo))
    if len(shared) != 42:
        raise SystemExit(f"expected 42 paired scores, found {len(shared)}")
    base_values = np.array([float(base[key]["rubric_reward"]) for key in shared])
    grpo_values = np.array([float(grpo[key]["rubric_reward"]) for key in shared])
    delta = grpo_values - base_values
    rng = np.random.default_rng(20260918)
    indices = rng.integers(0, len(shared), size=(args.bootstrap, len(shared)))
    boot = delta[indices].mean(axis=1)
    report = {
        "paired_tasks": len(shared),
        "base_skill_mean": float(base_values.mean()),
        "grpo_skill_mean": float(grpo_values.mean()),
        "paired_mean_delta": float(delta.mean()),
        "bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "positive_tasks": int((delta > 0).sum()),
        "tie_tasks": int((delta == 0).sum()),
        "negative_tasks": int((delta < 0).sum()),
    }
    if args.no_skill:
        no_skill = last_ok(Path(args.no_skill))
        if set(no_skill) != set(base):
            raise SystemExit("No-Skill and Base+Skill task sets do not match")
        no_skill_values = np.array([float(no_skill[key]["rubric_reward"]) for key in shared])
        report["base_no_skill_mean"] = float(no_skill_values.mean())
        report["base_skill_lift"] = float(base_values.mean() - no_skill_values.mean())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
