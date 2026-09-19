#!/usr/bin/env python3
"""Re-run structural verifier failures while preserving complete raw generations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvr.judge import JudgeResult, TokenPlanJudge, _extract_json, weighted_rubric_reward

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "artifacts/qwen4b_verifier_bad_cases_001/trajectories.jsonl"
OUTPUT = ROOT / "artifacts/qwen4b_verifier_bad_cases_001/raw_protocol_reproductions.jsonl"
MODEL_PATH = ROOT.parent / "models/Qwen3-4B-Instruct-2507"


def attempt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    rubric: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
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
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
    result: dict[str, Any] = {
        "budget": budget,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion_ids.shape[-1]),
        "finish_reason": "length" if int(completion_ids.shape[-1]) >= budget else "stop",
        "elapsed_seconds": round(elapsed, 3),
        "raw_judge_output": raw,
    }
    try:
        parsed = _extract_json(raw)
        result["parsed_json"] = parsed
        validated = JudgeResult.model_validate(parsed)
        result["validated_judge"] = validated.model_dump()
        result["rubric_reward"] = weighted_rubric_reward(validated, rubric)
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - preserve arbitrary Judge failures for audit
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def main() -> int:
    cases = [
        json.loads(line)
        for line in CASES.read_text().splitlines()
        if line.strip() and json.loads(line)["category"] == "structural_protocol_failure"
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    with OUTPUT.open("w", encoding="utf-8", buffering=1) as handle:
        for index, case in enumerate(cases, 1):
            task = case["task"]
            instruction = TokenPlanJudge.build_instruction(
                task["task_prompt"], case["policy_rollout"]["response"], task["rubric"]
            )
            first = attempt(model, tokenizer, instruction, task["rubric"], 1024)
            attempts = [first]
            if first["status"] == "error" and first["finish_reason"] == "length":
                attempts.append(attempt(model, tokenizer, instruction, task["rubric"], 2048))
            output = {
                "key": case["key"],
                "original_error": case["qwen_verifier"],
                "system_prompt": "Return only valid JSON. Do not use markdown.",
                "complete_judge_instruction": instruction,
                "attempts": attempts,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            final = attempts[-1]
            print(
                f"[{index}/{len(cases)}] {case['key']['task_id']} "
                f"status={final['status']} finish={final['finish_reason']} "
                f"tokens={final['completion_tokens']} error={final.get('error')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
