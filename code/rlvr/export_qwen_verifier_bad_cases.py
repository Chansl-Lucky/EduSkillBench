#!/usr/bin/env python3
"""Export auditable Qwen verifier failures and high-disagreement trajectories."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts/qwen4b_verifier_bad_cases_001"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_by_key(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (record["task_id"], record["condition"], int(record["seed"]))
        result[key] = record
    return result


def load_official_tasks(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rubric = json.loads(row["rubric"])
            result[row["task_id"]] = {
                "task_id": row["task_id"],
                "skill_id": row["skill_id"],
                "context": row["context"],
                "user_prompt": row["user_prompt"],
                "task_prompt": f"{row['context'].strip()}\n\n{row['user_prompt'].strip()}",
                "expected_output": row["expected_output"],
                "rubric": rubric,
            }
    return result


def prompt_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for item in read_jsonl(path):
        item["task_prompt"] = f"{item['context'].strip()}\n\n{item['user_prompt'].strip()}"
        result[item["task_id"]] = item
    return result


def compact_score(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "judge_model": record.get("judge_model"),
        "judge_protocol": record.get("judge_protocol"),
        "status": record.get("status"),
        "rubric_reward": record.get("rubric_reward"),
        "reward": record.get("reward"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
        "judge": record.get("judge"),
    }


def trajectory(
    *,
    category: str,
    task: dict[str, Any],
    rollout: dict[str, Any],
    qwen: dict[str, Any] | None,
    deepseek: dict[str, Any] | None,
    note: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "note": note,
        "key": {
            "task_id": rollout["task_id"],
            "condition": rollout["condition"],
            "seed": rollout["seed"],
        },
        "task": {
            "context": task["context"],
            "user_prompt": task["user_prompt"],
            "task_prompt": task["task_prompt"],
            "expected_output": task.get("expected_output"),
            "rubric": task["rubric"],
        },
        "policy_rollout": rollout,
        "qwen_verifier": compact_score(qwen),
        "deepseek_comparator": compact_score(deepseek),
    }


def md_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def main() -> int:
    official_tasks = load_official_tasks(ROOT / "data/single_turn_tasks.csv")
    calibration_tasks = prompt_map(ROOT / "artifacts/reward_feasibility_001/prompts.jsonl")

    official_rollouts = last_by_key(
        read_jsonl(ROOT / "artifacts/overnight_dsflash_20260914/rollouts.jsonl")
    )
    calibration_rollouts = last_by_key(
        read_jsonl(ROOT / "artifacts/reward_feasibility_001/rollouts.jsonl")
    )
    official_qwen = last_by_key(
        read_jsonl(ROOT / "artifacts/qwen4b_skill_baseline_rescore_001/scores.jsonl")
    )
    calibration_qwen = last_by_key(
        read_jsonl(ROOT / "artifacts/qwen4b_verifier_calibration_001/scores.jsonl")
    )
    calibration_ds = last_by_key(
        read_jsonl(
            ROOT
            / "artifacts/reward_feasibility_002/frozen/dense_scores_before_qwen_20260917T110323Z.jsonl"
        )
    )

    cases: list[dict[str, Any]] = []

    # Every deterministic protocol failure from the latest record for each baseline cell.
    for key, score in sorted(official_qwen.items()):
        if score.get("status") != "ok":
            cases.append(
                trajectory(
                    category="structural_protocol_failure",
                    task=official_tasks[key[0]],
                    rollout=official_rollouts[key],
                    qwen=score,
                    deepseek=None,
                    note="Qwen output was generated but could not be converted into a complete rubric reward.",
                )
            )

    # Every deterministic calibration protocol failure.
    for key, score in sorted(calibration_qwen.items()):
        if score.get("status") != "ok":
            cases.append(
                trajectory(
                    category="structural_protocol_failure",
                    task=calibration_tasks[key[0]],
                    rollout=calibration_rollouts[key],
                    qwen=score,
                    deepseek=calibration_ds.get(key),
                    note="Qwen output was generated but could not be converted into a complete rubric reward.",
                )
            )

    # Top same-protocol disagreements expose false/over-confident full scores.
    disagreements = []
    for key, qwen in calibration_qwen.items():
        ds = calibration_ds.get(key)
        if qwen.get("status") != "ok" or not ds or ds.get("status") != "ok":
            continue
        delta = float(qwen["rubric_reward"]) - float(ds["rubric_reward"])
        disagreements.append((delta, key, qwen, ds))
    disagreements.sort(reverse=True)
    for delta, key, qwen, ds in disagreements[:8]:
        cases.append(
            trajectory(
                category="semantic_overcredit_disagreement",
                task=calibration_tasks[key[0]],
                rollout=calibration_rollouts[key],
                qwen=qwen,
                deepseek=ds,
                note=(
                    f"Same continuous rubric protocol: Qwen minus DeepSeek = {delta:+.3f}. "
                    "This is a disagreement signal, not proof that DeepSeek is ground truth."
                ),
            )
        )

    # One complete four-rollout group with Qwen collapse and the largest DS range.
    groups: dict[str, list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for key, qwen in calibration_qwen.items():
        ds = calibration_ds.get(key)
        if qwen.get("status") == "ok" and ds and ds.get("status") == "ok":
            groups[key[0]].append((key, qwen, ds))
    collapsed = []
    for task_id, items in groups.items():
        if len(items) != 4:
            continue
        qvals = [float(x[1]["rubric_reward"]) for x in items]
        dvals = [float(x[2]["rubric_reward"]) for x in items]
        if max(qvals) == min(qvals):
            collapsed.append((max(dvals) - min(dvals), task_id, items))
    if collapsed:
        _, _, items = max(collapsed)
        for key, qwen, ds in sorted(items):
            cases.append(
                trajectory(
                    category="collapsed_four_candidate_group",
                    task=calibration_tasks[key[0]],
                    rollout=calibration_rollouts[key],
                    qwen=qwen,
                    deepseek=ds,
                    note="Qwen assigned the same reward to all four candidates in this GRPO-style group.",
                )
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    structural = [x for x in cases if x["category"] == "structural_protocol_failure"]
    disagreements_out = [x for x in cases if x["category"] == "semantic_overcredit_disagreement"]
    collapsed_out = [x for x in cases if x["category"] == "collapsed_four_candidate_group"]

    lines = [
        "# Qwen3-4B Verifier Bad-case Trajectories",
        "",
        "## Bottom line",
        "",
        (
            "The local verifier process completed, but Qwen3-4B is **not approved as the GRPO reward model**. "
            "Its main failure is semantic discrimination (ceiling/ties), not infrastructure. Some generations also "
            "fail the strict output contract by omitting a rubric item or returning truncated JSON."
        ),
        "",
        (
            "This report contains the complete task prompt, policy response, rubric, parsed Qwen judgement, and—where "
            "available—the same-protocol DeepSeek judgement. The DeepSeek comparison is diagnostic rather than ground truth."
        ),
        "",
        "## Case index",
        "",
        f"- Structural protocol failures: {len(structural)}",
        f"- Largest same-protocol Qwen/DeepSeek disagreements: {len(disagreements_out)}",
        f"- Four-candidate collapsed group trajectories: {len(collapsed_out)}",
        "",
    ]
    for index, case in enumerate(cases, 1):
        key = case["key"]
        q = case["qwen_verifier"] or {}
        d = case["deepseek_comparator"] or {}
        lines.extend(
            [
                f"## {index}. {case['category']}: `{key['task_id']}` / `{key['condition']}` / `{key['seed']}`",
                "",
                case["note"],
                "",
                (
                    f"Qwen status/reward: `{q.get('status')}` / `{q.get('rubric_reward')}`; "
                    f"DeepSeek status/reward: `{d.get('status')}` / `{d.get('rubric_reward')}`."
                ),
                "",
                "### Task and rubric",
                "",
                md_json(case["task"]),
                "",
                "### Complete policy response",
                "",
                str(case["policy_rollout"].get("response", "")),
                "",
                "### Qwen verifier result",
                "",
                md_json(case["qwen_verifier"]),
                "",
                "### DeepSeek comparator result",
                "",
                md_json(case["deepseek_comparator"]),
                "",
            ]
        )
    (OUT_DIR / "BAD_CASES.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUT_DIR), "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
