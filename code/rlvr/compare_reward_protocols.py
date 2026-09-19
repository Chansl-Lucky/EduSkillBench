#!/usr/bin/env python3
"""Compare dense rubric rewards with the frozen paper PASS/FAIL audit scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-scores", required=True)
    parser.add_argument("--audit-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-range", type=float, default=0.2)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def successful_by_cell(
    path: str | Path, protocol: str
) -> dict[tuple[str, str, int], dict[str, Any]]:
    chosen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "ok" or row.get("judge_protocol") != protocol:
            continue
        key = (row["task_id"], row["condition"], int(row["seed"]))
        chosen[key] = row
    return chosen


def safe_stat(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def main() -> int:
    args = parse_args()
    dense = successful_by_cell(args.dense_scores, "rubric_continuous")
    audit = successful_by_cell(args.audit_scores, "paper_pass_fail")
    shared = sorted(dense.keys() & audit.keys())
    if not shared:
        raise SystemExit("dense and audit score files have no successful shared cells")

    dense_values = [float(dense[key]["rubric_reward"]) for key in shared]
    audit_values = [float(audit[key]["rubric_reward"]) for key in shared]
    pearson = pearsonr(dense_values, audit_values)
    spearman = spearmanr(dense_values, audit_values)

    by_task: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for key in shared:
        by_task[key[0]].append(key)
    group_rows: list[dict[str, Any]] = []
    for task_id, keys in sorted(by_task.items()):
        dense_group = [float(dense[key]["rubric_reward"]) for key in keys]
        audit_group = [float(audit[key]["rubric_reward"]) for key in keys]
        dense_range = max(dense_group) - min(dense_group)
        audit_range = max(audit_group) - min(audit_group)
        group_rows.append(
            {
                "task_id": task_id,
                "cells": len(keys),
                "dense_mean": sum(dense_group) / len(dense_group),
                "audit_mean": sum(audit_group) / len(audit_group),
                "dense_range": dense_range,
                "audit_range": audit_range,
                "dense_unique_scores": len(set(dense_group)),
                "audit_unique_scores": len(set(audit_group)),
                "dense_nonzero_range": dense_range > 1e-9,
                "audit_nonzero_range": audit_range > 1e-9,
                "dense_nontrivial_range": dense_range >= args.minimum_range,
                "audit_nontrivial_range": audit_range >= args.minimum_range,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "protocol_comparison_groups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    group_count = len(group_rows)
    dense_nontrivial = sum(bool(row["dense_nontrivial_range"]) for row in group_rows)
    audit_nontrivial = sum(bool(row["audit_nontrivial_range"]) for row in group_rows)
    comparison = {
        "shared_cells": len(shared),
        "groups": group_count,
        "cell_level_pearson": safe_stat(float(pearson.statistic)),
        "cell_level_spearman": safe_stat(float(spearman.statistic)),
        "dense_nonzero_group_fraction": sum(bool(row["dense_nonzero_range"]) for row in group_rows)
        / group_count,
        "audit_nonzero_group_fraction": sum(bool(row["audit_nonzero_range"]) for row in group_rows)
        / group_count,
        "dense_nontrivial_group_fraction": dense_nontrivial / group_count,
        "audit_nontrivial_group_fraction": audit_nontrivial / group_count,
        "dense_all_tie_group_fraction": sum(
            int(row["dense_unique_scores"]) == 1 for row in group_rows
        )
        / group_count,
        "audit_all_tie_group_fraction": sum(
            int(row["audit_unique_scores"]) == 1 for row in group_rows
        )
        / group_count,
        "dense_variance_gate_at_least_0_70": dense_nontrivial / group_count >= 0.70,
        "paper_pass_fail_retained_as_audit": True,
    }
    (output_dir / "protocol_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
