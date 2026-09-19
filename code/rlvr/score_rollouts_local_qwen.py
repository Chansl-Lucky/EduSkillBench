#!/usr/bin/env python3
"""Score frozen rollout records with a local Qwen rubric verifier.

This is a resumable, single-GPU alternative to the HTTP Judge path.  It keeps
the same continuous-rubric prompt, parser, hard-reward mixture, and output
record shape used by ``score_rollouts.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvr.data import PromptRecord, load_official_tasks
from rlvr.judge import (
    JudgeProtocolError,
    JudgeResult,
    TokenPlanJudge,
    _extract_json,
    weighted_rubric_reward,
)
from rlvr.prompt_pool import load_prompt_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", default="qwen3-4b-instruct-2507-cdbee75-local")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens-cap", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_key(record: dict[str, Any], judge_model: str) -> tuple[str, str, int, str, str]:
    return (
        str(record["task_id"]),
        str(record["condition"]),
        int(record["seed"]),
        judge_model,
        "rubric_continuous",
    )


def rubric_payload(task: PromptRecord) -> list[dict[str, Any]]:
    return [item.model_dump() for item in task.rubric]


def generate_json(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    initial_budget: int,
    cap: int,
) -> tuple[JudgeResult, int, int, float, str]:
    budget = initial_budget
    last_text = ""
    while True:
        messages = [
            {"role": "system", "content": "Return only valid JSON. Do not use markdown."},
            {"role": "user", "content": instruction},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.monotonic() - started
        completion_ids = generated[0, prompt_tokens:]
        last_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        finish_reason = "length" if int(completion_ids.shape[-1]) >= budget else "stop"
        try:
            parsed = _extract_json(last_text)
            result = JudgeResult.model_validate(parsed)
            return result, prompt_tokens, int(completion_ids.shape[-1]), elapsed, finish_reason
        except (JudgeProtocolError, ValueError):
            if finish_reason == "length" and budget < cap:
                budget = min(cap, budget * 2)
                continue
            raise JudgeProtocolError(
                f"local Qwen returned invalid rubric JSON: {last_text[:240]!r}"
            )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.max_new_tokens <= 0 or args.max_new_tokens_cap < args.max_new_tokens:
        raise SystemExit("invalid generation token budgets")

    rollouts = read_jsonl(Path(args.input))
    if args.limit is not None:
        rollouts = rollouts[: args.limit]
    task_path = Path(args.tasks)
    task_records = (
        load_prompt_pool(task_path)
        if task_path.suffix == ".jsonl"
        else load_official_tasks(task_path)
    )
    tasks = {task.task_id: task for task in task_records}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str, int, str, str]] = set()
    if output.exists():
        for record in read_jsonl(output):
            if record.get("status") == "ok":
                completed.add(score_key(record, str(record["judge_model"])))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    with output.open("a", buffering=1) as handle:
        for index, rollout in enumerate(rollouts, start=1):
            key = score_key(rollout, args.judge_model)
            if key in completed:
                continue
            task = tasks.get(str(rollout["task_id"]))
            if task is None:
                raise ValueError(f"unknown task_id: {rollout['task_id']}")
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
            rubric = rubric_payload(task)
            base = {
                **rollout,
                "judge_model": args.judge_model,
                "rubric_criteria": len(rubric),
                "judge_protocol": "rubric_continuous",
                "truncated": truncated,
            }
            try:
                result, prompt_tokens, completion_tokens, elapsed, finish_reason = generate_json(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=TokenPlanJudge.build_instruction(task.question, response, rubric),
                    initial_budget=args.max_new_tokens,
                    cap=args.max_new_tokens_cap,
                )
                rubric_reward = weighted_rubric_reward(result, rubric)
                reward = max(
                    0.0,
                    min(
                        1.0,
                        0.85 * rubric_reward + 0.15 * hard_reward - failure_penalty,
                    ),
                )
                result.judge_model = args.judge_model
                result.max_tokens_used = args.max_new_tokens
                result.finish_reason = finish_reason
                result.usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                record = {
                    **base,
                    "status": "ok",
                    "reward": reward,
                    "rubric_reward": rubric_reward,
                    "hard_reward": hard_reward,
                    "failure_penalty": failure_penalty,
                    "judge": result.model_dump(),
                    "judge_elapsed_seconds": round(elapsed, 3),
                }
            except Exception as exc:  # noqa: BLE001 - keep failures auditable and resumable
                record = {
                    **base,
                    "status": "error",
                    "reward": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(rollouts)}] {rollout['task_id']} seed={rollout['seed']} "
                f"status={record['status']} reward={record.get('reward')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
