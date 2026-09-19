#!/usr/bin/env python3
"""Prepare frozen Base+Skill rollout cells for a Base-vs-GRPO comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=("no_skill", "skill_text"), default="skill_text")
    args = parser.parse_args()
    records = [
        json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()
    ]
    selected = [record for record in records if record.get("condition") == args.condition]
    by_task = {record["task_id"]: record for record in selected}
    if len(by_task) != 42:
        raise SystemExit(f"expected 42 unique {args.condition} cells, found {len(by_task)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(by_task[key], ensure_ascii=False) + "\n" for key in sorted(by_task)),
        encoding="utf-8",
    )
    print(f"prepared={len(by_task)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
