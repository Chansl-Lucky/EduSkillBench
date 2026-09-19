import json

import pytest
from rlvr.data import PromptRecord, load_official_tasks
from rlvr.prompt_pool import (
    assert_disjoint,
    assert_semantically_separated,
    load_prompt_pool,
    semantic_overlap_report,
)


def _item(task_id: str, split: str = "train") -> dict:
    return {
        "task_id": task_id,
        "skill_id": "test-skill",
        "split": split,
        "scenario": "Teaching Material Generation",
        "subject": "Science",
        "education_level": "Grade 8",
        "difficulty": "easy",
        "context": "A fresh context.",
        "user_prompt": f"Create a new educational artifact for {task_id}.",
        "rubric": [
            {"criterion": "A", "points": 60, "description": "first"},
            {"criterion": "B", "points": 40, "description": "second"},
        ],
    }


def test_pool_rejects_official_split(tmp_path) -> None:
    path = tmp_path / "pool.jsonl"
    path.write_text(json.dumps(_item("x", "official_test")) + "\n")
    with pytest.raises(ValueError, match="split"):
        load_prompt_pool(path)


def test_pools_are_checked_for_leakage(tmp_path) -> None:
    path = tmp_path / "pool.jsonl"
    path.write_text(json.dumps(_item("x")) + "\n")
    pool = load_prompt_pool(path)
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(pool, pool)


def test_official_tasks_are_not_train_pool() -> None:
    official = load_official_tasks("data/single_turn_tasks.csv")
    assert all(item.split == "official_test" for item in official)


def test_semantic_overlap_flags_lightly_reworded_prompt() -> None:
    reference = PromptRecord.model_validate(
        {
            **_item("reference", "official_test"),
            "context": "Students have learned photosynthesis reactants and products.",
            "user_prompt": "Create a four-option diagnostic checkpoint on photosynthesis.",
            "source": "reference",
        }
    )
    candidate = PromptRecord.model_validate(
        {
            **_item("candidate", "judge_calibration"),
            "context": "The students learned photosynthesis products and reactants.",
            "user_prompt": "Design a four-option diagnostic checkpoint about photosynthesis.",
            "source": "candidate",
        }
    )
    report = semantic_overlap_report([candidate], [reference])
    assert report[0]["similarity"] > 0.72
    with pytest.raises(ValueError, match="near-duplicate"):
        assert_semantically_separated([candidate], [reference])
