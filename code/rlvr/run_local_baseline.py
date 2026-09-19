#!/usr/bin/env python3
"""Generate resumable local-policy baselines from official or independent tasks.

This is a weight-level policy baseline, not a reproduction of the OpenCode
agent harness. The with-skill condition injects the released SKILL.md text into
the system context and is therefore labelled ``skill_text`` in every record.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rlvr.data import build_messages, load_official_tasks, load_skill_text
from rlvr.prompt_pool import load_prompt_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Optional PEFT adapter to load on the base model")
    parser.add_argument("--tasks", default="data/single_turn_tasks.csv")
    parser.add_argument("--skill-root", default="skills/single_turn")
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=["no_skill", "skill_text", "both"], default="both")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Only run this task ID; repeat the option to select multiple tasks",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        completed.add((item["task_id"], item["condition"], int(item["seed"])))
    return completed


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this baseline runner")

    set_seed(args.seed)
    random.seed(args.seed)
    task_path = Path(args.tasks)
    tasks = (
        load_prompt_pool(task_path)
        if task_path.suffix == ".jsonl"
        else load_official_tasks(task_path)
    )
    if args.task_ids:
        selected = set(args.task_ids)
        tasks = [task for task in tasks if task.task_id in selected]
        missing = selected - {task.task_id for task in tasks}
        if missing:
            raise SystemExit(f"unknown --task-id values: {sorted(missing)}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    conditions = ["no_skill", "skill_text"] if args.condition == "both" else [args.condition]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    if args.temperature <= 0:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output)

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for task in tasks:
            for condition in conditions:
                key = (task.task_id, condition, args.seed)
                if key in completed:
                    continue
                skill_text = (
                    load_skill_text(args.skill_root, task.skill_id)
                    if condition == "skill_text"
                    else None
                )
                messages = build_messages(task, skill_text)
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                prompt_tokens = int(encoded["input_ids"].shape[-1])
                started = time.monotonic()
                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if args.temperature > 0:
                    generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        **generation_kwargs,
                    )
                elapsed = time.monotonic() - started
                completion_ids = generated[0, prompt_tokens:]
                response = tokenizer.decode(completion_ids, skip_special_tokens=True)
                ended_with_eos = bool(
                    completion_ids.numel()
                    and int(completion_ids[-1]) == int(tokenizer.eos_token_id)
                )
                record = {
                    "task_id": task.task_id,
                    "skill_id": task.skill_id,
                    "split": task.split,
                    "condition": condition,
                    "model": str(Path(args.model).resolve()),
                    "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": int(completion_ids.shape[-1]),
                    "truncated": bool(
                        completion_ids.shape[-1] >= args.max_new_tokens and not ended_with_eos
                    ),
                    "elapsed_seconds": round(elapsed, 3),
                    "response": response,
                    "prompt_fingerprint": task.normalized_fingerprint,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"{task.task_id} {condition}: prompt={prompt_tokens} "
                    f"completion={record['completion_tokens']} time={elapsed:.1f}s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
