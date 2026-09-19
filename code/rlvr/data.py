"""Load released EduSkillBench tasks without mutating the official data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class RubricCriterion(BaseModel):
    criterion: str
    points: float = Field(gt=0)
    description: str


class PromptRecord(BaseModel):
    task_id: str
    skill_id: str
    split: str
    scenario: str
    subject: str
    education_level: str
    difficulty: str
    context: str
    user_prompt: str
    expected_output: str | None = None
    rubric: list[RubricCriterion]
    source: str

    @model_validator(mode="after")
    def validate_rubric_total(self) -> PromptRecord:
        total = sum(item.points for item in self.rubric)
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"rubric points must total 100, got {total}")
        return self

    @property
    def question(self) -> str:
        return f"{self.context.strip()}\n\n{self.user_prompt.strip()}"

    @property
    def normalized_fingerprint(self) -> str:
        normalized = re.sub(r"\W+", "", self.question.casefold())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_official_tasks(path: str | Path) -> list[PromptRecord]:
    path = Path(path)
    records: list[PromptRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                PromptRecord(
                    task_id=row["task_id"],
                    skill_id=row["skill_id"],
                    split="official_test",
                    scenario=row["edubench_scenario"],
                    subject=row["subject"],
                    education_level=row["education_level"],
                    difficulty=row["difficulty"],
                    context=row["context"],
                    user_prompt=row["user_prompt"],
                    expected_output=row["expected_output"],
                    rubric=[
                        RubricCriterion.model_validate(item) for item in json.loads(row["rubric"])
                    ],
                    source=str(path),
                )
            )
    return records


def load_skill_text(skill_root: str | Path, skill_id: str) -> str:
    path = Path(skill_root) / skill_id / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"released skill not found: {path}")
    return path.read_text(encoding="utf-8")


def build_messages(task: PromptRecord, skill_text: str | None = None) -> list[dict[str, str]]:
    system = (
        "You are an educational assistant. Complete the requested educational artifact "
        "carefully, accurately, and in a form the teacher can use directly."
    )
    if skill_text is not None:
        system += (
            "\n\nThe following is the matched reusable educational Skill. Apply its procedure "
            "to the task. Treat examples as guidance, not as content to copy.\n\n"
            "<skill>\n"
            f"{skill_text.strip()}\n"
            "</skill>"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task.question},
    ]
