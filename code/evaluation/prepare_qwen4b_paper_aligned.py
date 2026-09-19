#!/usr/bin/env python3
"""Create an immutable-by-convention copy of the 42 official eval inputs.

The released skills are never edited.  Only evaluation defaults are changed:
one timeout for both arms and the exact local model alias as primary judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("skills/single_turn"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper_aligned_qwen3_4b_001/input_skills"),
    )
    parser.add_argument("--judge-model", default="qwen3-4b-instruct-2507-cdbee75")
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive")
    skill_dirs = sorted(p.parent.parent for p in args.source.glob("*/evals/evals.json"))
    if len(skill_dirs) != 14:
        raise SystemExit(f"Expected 14 released single-turn skills, found {len(skill_dirs)}")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite prepared inputs: {args.output}")

    args.output.mkdir(parents=True)
    manifest: dict[str, object] = {
        "judge_model": args.judge_model,
        "timeout_sec": args.timeout_sec,
        "skills": [],
    }
    total_cases = 0
    for source_dir in skill_dirs:
        target_dir = args.output / source_dir.name
        target_dir.mkdir()
        shutil.copy2(source_dir / "SKILL.md", target_dir / "SKILL.md")
        (target_dir / "evals").mkdir()

        source_eval = source_dir / "evals" / "evals.json"
        payload = json.loads(source_eval.read_text())
        payload.setdefault("defaults", {})["timeout_sec"] = args.timeout_sec
        payload["defaults"]["judge_model"] = args.judge_model
        target_eval = target_dir / "evals" / "evals.json"
        target_eval.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        case_count = len(payload.get("cases", []))
        total_cases += case_count
        manifest["skills"].append(
            {
                "skill_id": source_dir.name,
                "cases": case_count,
                "source_eval_sha256": sha256(source_eval),
                "prepared_eval_sha256": sha256(target_eval),
                "skill_sha256": sha256(source_dir / "SKILL.md"),
            }
        )

    if total_cases != 42:
        shutil.rmtree(args.output)
        raise SystemExit(f"Expected 42 cases, found {total_cases}; removed incomplete output")
    manifest["skill_count"] = len(skill_dirs)
    manifest["case_count"] = total_cases
    manifest_path = args.output.parent / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Prepared {len(skill_dirs)} skills / {total_cases} cases at {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
