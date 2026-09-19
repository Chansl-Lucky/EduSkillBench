#!/usr/bin/env python3
"""Assemble validated per-Skill JSONL candidates into one prompt pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.data import load_official_tasks
from rlvr.prompt_pool import (
    assert_disjoint,
    assert_semantically_separated,
    load_prompt_pool,
    semantic_overlap_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--official", default="data/single_turn_tasks.csv")
    parser.add_argument("--expected-files", type=int, default=14)
    parser.add_argument("--expected-records", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(Path(args.input_dir).glob("*.jsonl"))
    if len(paths) != args.expected_files:
        raise SystemExit(f"expected {args.expected_files} pool files, found {len(paths)}")
    pools = [load_prompt_pool(path) for path in paths]
    candidates = [record for pool in pools for record in pool]
    if len(candidates) != args.expected_records:
        raise SystemExit(f"expected {args.expected_records} records, found {len(candidates)}")
    official = load_official_tasks(args.official)
    assert_disjoint(*pools, official)
    assert_semantically_separated(candidates, official)
    overlap = semantic_overlap_report(candidates, official)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record.model_dump(), ensure_ascii=False) + "\n" for record in candidates
        ),
        encoding="utf-8",
    )
    load_prompt_pool(temporary)
    temporary.replace(output)

    report = {
        "pool_files": len(paths),
        "records": len(candidates),
        "skills": len({record.skill_id for record in candidates}),
        "splits": sorted({record.split for record in candidates}),
        "maximum_official_similarity": max(float(row["similarity"]) for row in overlap),
        "closest_official_pairs": sorted(
            overlap, key=lambda row: float(row["similarity"]), reverse=True
        )[:10],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
