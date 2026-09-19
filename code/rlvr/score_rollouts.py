#!/usr/bin/env python3
"""Score local rollout JSONL records with a frozen TokenPlan rubric Judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rlvr.data import PromptRecord, load_official_tasks
from rlvr.judge import TokenPlanJudge
from rlvr.prompt_pool import load_prompt_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="data/single_turn_tasks.csv")
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Exact model ID from the provider's current /models response",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TOKENPLAN_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1"),
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens-cap", type=int, default=8192)
    parser.add_argument(
        "--judge-protocol",
        choices=("rubric_continuous", "paper_pass_fail"),
        default="rubric_continuous",
    )
    parser.add_argument("--limit", type=int, help="Score only the first N rollout records")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def score_key(
    record: dict[str, Any], judge_model: str, judge_protocol: str
) -> tuple[str, str, int, str, str]:
    return (
        record["task_id"],
        record["condition"],
        int(record["seed"]),
        judge_model,
        judge_protocol,
    )


def rubric_payload(task: PromptRecord) -> list[dict[str, Any]]:
    return [item.model_dump() for item in task.rubric]


async def main_async() -> int:
    args = parse_args()
    api_key = os.environ.get("TOKENPLAN_API_KEY")
    if not api_key:
        raise SystemExit("TOKENPLAN_API_KEY is required")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rollouts = read_jsonl(input_path)
    if args.limit is not None:
        rollouts = rollouts[: args.limit]
    task_path = Path(args.tasks)
    task_records = (
        load_prompt_pool(task_path)
        if task_path.suffix == ".jsonl"
        else load_official_tasks(task_path)
    )
    tasks = {task.task_id: task for task in task_records}

    completed: set[tuple[str, str, int, str, str]] = set()
    if output_path.exists():
        for record in read_jsonl(output_path):
            if record.get("status") == "ok":
                completed.add(
                    score_key(
                        record,
                        record["judge_model"],
                        record.get("judge_protocol", "rubric_continuous"),
                    )
                )

    judge = TokenPlanJudge(
        api_key=api_key,
        model=args.judge_model,
        base_url=args.base_url,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_tokens_cap=args.max_tokens_cap,
        protocol=args.judge_protocol,
    )
    visible_models = await judge.list_models()
    if args.judge_model not in visible_models:
        await judge.aclose()
        raise SystemExit(
            f"judge model {args.judge_model!r} is unavailable; visible models: {visible_models}"
        )
    write_lock = asyncio.Lock()

    async def score_one(rollout: dict[str, Any], handle: Any) -> None:
        key = score_key(rollout, args.judge_model, args.judge_protocol)
        if key in completed:
            return
        task = tasks.get(rollout["task_id"])
        if task is None:
            raise ValueError(f"unknown task_id in rollout: {rollout['task_id']}")
        response = str(rollout.get("response") or "")
        inferred_truncated = int(rollout.get("completion_tokens", 0)) >= int(
            rollout.get("max_new_tokens", 0)
        )
        truncated = bool(rollout.get("truncated", inferred_truncated))
        if not response.strip():
            hard_reward, failure_penalty = 0.0, 1.0
        elif truncated:
            hard_reward, failure_penalty = 0.5, 0.1
        else:
            hard_reward, failure_penalty = 1.0, 0.0
        base = {
            **rollout,
            "judge_model": args.judge_model,
            "rubric_criteria": len(task.rubric),
            "judge_protocol": args.judge_protocol,
            "truncated": truncated,
        }
        try:
            result = await judge.score_reward(
                task.question,
                response,
                rubric_payload(task),
                expected_output=task.expected_output,
                hard_reward=hard_reward,
                failure_penalty=(
                    0.0 if args.judge_protocol == "paper_pass_fail" else failure_penalty
                ),
                hard_weight=0.0 if args.judge_protocol == "paper_pass_fail" else 0.15,
            )
            record = {
                **base,
                "status": "ok",
                "reward": result.reward,
                "rubric_reward": result.rubric_reward,
                "hard_reward": result.hard_reward,
                "failure_penalty": result.failure_penalty,
                "judge": result.result.model_dump(),
            }
        # A failed Judge call must remain a missing reward instead of aborting
        # the batch or being silently converted to a low reward.
        except Exception as exc:  # noqa: BLE001
            record = {
                **base,
                "status": "error",
                "reward": None,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
        async with write_lock:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{rollout['task_id']} {rollout['condition']} "
                f"judge={args.judge_model} status={record['status']} reward={record.get('reward')}",
                flush=True,
            )

    try:
        # Each append is one small record; keeping writes serialized makes the
        # JSONL resumable without adding an async filesystem dependency.
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:  # noqa: ASYNC230
            await asyncio.gather(*(score_one(rollout, handle) for rollout in rollouts))
    finally:
        await judge.aclose()
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
