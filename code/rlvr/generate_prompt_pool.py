#!/usr/bin/env python3
"""Generate independent RL/calibration prompts from released Skill definitions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from rlvr.data import PromptRecord, load_official_tasks
from rlvr.prompt_pool import assert_disjoint, assert_semantically_separated, load_prompt_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--split", choices=("train", "dev", "judge_calibration"), required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="data/single_turn_tasks.csv")
    parser.add_argument("--skills", default="skills/single_turn")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TOKENPLAN_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1"),
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    return parser.parse_args()


def extract_array(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("generation did not contain a JSON array")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("generation JSON must be an array of objects")
    return value


def build_instruction(
    *,
    skill_id: str,
    skill_text: str,
    scenario: str,
    count: int,
    forbidden_topics: list[str],
) -> str:
    return f"""
Create exactly {count} NEW prompts for calibrating and training a rubric-guided educational
assistant. They must exercise the procedure in the supplied Skill, but must not copy its examples
or any released benchmark topic. The assistant response is NOT requested: create tasks only.

Metadata skill_id: {skill_id}
Scenario: {scenario}
Released topics that are forbidden: {json.dumps(forbidden_topics, ensure_ascii=False)}

Rules:
- English only; do not mention the Skill, benchmark, rubric, or evaluation in user-facing text.
- Use subjects, concepts, learner needs, settings, and surface forms distinct from forbidden topics.
- Make each prompt genuinely different in topic and pedagogical decision, not a name swap.
- A capable model without the Skill must still be able to attempt it.
- Rubric criteria must be observable in the response, use exact integer points totaling 100, and
  must not reward mentioning the Skill or copying terminology.
- Vary difficulty and education level. No web access or external files may be required.
- Return JSON only as an array of exactly {count} objects.

Object schema:
{{"subject":"", "education_level":"", "difficulty":"easy|medium|hard",
  "context":"", "user_prompt":"", "expected_output":"",
  "rubric":[{{"criterion":"", "points":20, "description":""}}]}}

SOURCE SKILL:
<skill>
{skill_text}
</skill>
""".strip()


async def request_generation(args: argparse.Namespace, instruction: str) -> list[dict[str, Any]]:
    api_key = os.environ.get("TOKENPLAN_API_KEY")
    if not api_key:
        raise SystemExit("TOKENPLAN_API_KEY is required")
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=300,
    ) as client:
        models_response = await client.get("/models")
        models_response.raise_for_status()
        models = {item["id"] for item in models_response.json().get("data", [])}
        if args.model not in models:
            raise SystemExit(
                f"generation model {args.model!r} unavailable; visible models: {sorted(models)}"
            )
        response = await client.post(
            "/chat/completions",
            json={
                "model": args.model,
                "messages": [
                    {"role": "system", "content": "Return only the requested valid JSON array."},
                    {"role": "user", "content": instruction},
                ],
                "temperature": 0.35,
                "max_tokens": args.max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if not content:
        raise RuntimeError(
            "generation returned no content "
            f"(finish_reason={choice.get('finish_reason')}, usage={data.get('usage')})"
        )
    return extract_array(content)


def materialize_records(
    raw_tasks: list[dict[str, Any]],
    *,
    skill_id: str,
    scenario: str,
    split: str,
    source: str,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for index, item in enumerate(raw_tasks, 1):
        records.append(
            PromptRecord.model_validate(
                {
                    **item,
                    "task_id": f"{split}__{skill_id}__{index:03d}",
                    "skill_id": skill_id,
                    "split": split,
                    "scenario": scenario,
                    "source": source,
                }
            )
        )
    return records


async def main_async() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    official = load_official_tasks(args.tasks)
    matched_official = [item for item in official if item.skill_id == args.skill_id]
    if not matched_official:
        raise SystemExit(f"unknown single-turn skill: {args.skill_id}")
    skill_path = Path(args.skills) / args.skill_id / "SKILL.md"
    if not skill_path.is_file():
        raise SystemExit(f"missing Skill file: {skill_path}")
    forbidden_topics = [f"{item.subject}: {item.context}" for item in matched_official]
    instruction = build_instruction(
        skill_id=args.skill_id,
        skill_text=skill_path.read_text(encoding="utf-8"),
        scenario=matched_official[0].scenario,
        count=args.count,
        forbidden_topics=forbidden_topics,
    )
    raw_tasks = await request_generation(args, instruction)
    if len(raw_tasks) != args.count:
        raise ValueError(f"expected {args.count} tasks, received {len(raw_tasks)}")
    records = materialize_records(
        raw_tasks,
        skill_id=args.skill_id,
        scenario=matched_official[0].scenario,
        split=args.split,
        source=f"generated:{args.model}",
    )
    assert_disjoint(records, official)
    assert_semantically_separated(records, official)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        raise FileExistsError(f"refusing to overwrite non-empty prompt pool: {output}")
    output.write_text(
        "".join(json.dumps(item.model_dump(), ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    load_prompt_pool(output)
    print(f"generated={len(records)} split={args.split} skill={args.skill_id} output={output}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
