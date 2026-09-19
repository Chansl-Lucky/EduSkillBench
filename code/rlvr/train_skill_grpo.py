#!/usr/bin/env python3
"""Run a small Skill-conditioned GRPO experiment with a remote rubric Judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rlvr.data import build_messages, load_skill_text
from rlvr.judge import TokenPlanJudge
from rlvr.prompt_pool import load_prompt_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--skills", default="skills/single_turn")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return str(completion)


def build_dataset(prompt_path: Path, skill_root: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    for task in load_prompt_pool(prompt_path):
        skill_text = load_skill_text(skill_root, task.skill_id)
        rows.append(
            {
                "prompt": build_messages(task, skill_text),
                "task_id": task.task_id,
                "task_prompt": task.question,
                "rubric_json": json.dumps(
                    [criterion.model_dump() for criterion in task.rubric], ensure_ascii=False
                ),
                "expected_output": task.expected_output or "",
            }
        )
    if not rows:
        raise ValueError("training prompt pool is empty")
    return Dataset.from_list(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("TOKENPLAN_API_KEY")
    if not api_key:
        raise SystemExit("ARK_API_KEY or TOKENPLAN_API_KEY is required")
    if args.num_generations < 2:
        raise SystemExit("GRPO requires at least two generations")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_trace = output_dir / "reward_trace.jsonl"
    dataset = build_dataset(Path(args.prompts), Path(args.skills))

    async def rubric_reward(
        completions: list[Any],
        task_id: list[str],
        task_prompt: list[str],
        rubric_json: list[str],
        expected_output: list[str],
        **_: Any,
    ) -> list[float]:
        judge = TokenPlanJudge(
            api_key=api_key,
            model=args.judge_model,
            base_url=args.base_url,
            concurrency=min(4, len(completions)),
            max_tokens=2048,
            max_tokens_cap=8192,
            protocol="rubric_continuous",
        )

        async def score_one(index: int) -> tuple[float, dict[str, Any]]:
            text = completion_text(completions[index])
            rubric = json.loads(rubric_json[index])
            if not text.strip():
                return 0.0, {
                    "task_id": task_id[index],
                    "completion": text,
                    "status": "empty",
                    "reward": 0.0,
                }
            result = await judge.score_reward(
                task_prompt[index],
                text,
                rubric,
                expected_output=expected_output[index],
                hard_reward=1.0,
                failure_penalty=0.0,
                hard_weight=0.15,
            )
            return result.reward, {
                "task_id": task_id[index],
                "completion": text,
                "status": "ok",
                "reward": result.reward,
                "rubric_reward": result.rubric_reward,
                "judge": result.result.model_dump(),
            }

        try:
            scored = await asyncio.gather(*(score_one(i) for i in range(len(completions))))
        finally:
            await judge.aclose()
        with reward_trace.open("a", encoding="utf-8") as handle:
            for _, record in scored:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        rewards = [float(reward) for reward, _ in scored]
        if max(rewards) == min(rewards):
            print(f"warning: zero-variance GRPO group reward={rewards[0]:.4f}", flush=True)
        else:
            print(
                f"GRPO reward group min={min(rewards):.4f} max={max(rewards):.4f} "
                f"range={max(rewards) - min(rewards):.4f}",
                flush=True,
            )
        return rewards

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = GRPOConfig(
        output_dir=str(output_dir / "checkpoint"),
        overwrite_output_dir=True,
        run_name=output_dir.name,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.9,
        top_p=0.9,
        learning_rate=1.0e-5,
        beta=0.02,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_first_step=True,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_generations if args.log_completions else None,
        save_strategy="steps",
        save_steps=min(args.save_steps, args.max_steps),
        save_total_limit=3,
        save_only_model=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "dtype": torch.bfloat16,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    started = time.time()
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=rubric_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    resume_checkpoint: str | None = None
    if args.resume:
        checkpoint_root = output_dir / "checkpoint"
        checkpoints = sorted(
            checkpoint_root.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if checkpoints:
            resume_checkpoint = str(checkpoints[-1])
            print(f"resuming from {resume_checkpoint}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    metrics = {
        **result.metrics,
        "elapsed_wall_seconds": round(time.time() - started, 3),
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "skill_conditioned": True,
        "train_prompts": len(dataset),
        "num_generations": args.num_generations,
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
