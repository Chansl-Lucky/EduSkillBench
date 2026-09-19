from pathlib import Path

from rlvr.data import build_messages, load_official_tasks, load_skill_text

ROOT = Path(__file__).parents[1]


def test_released_single_turn_invariants() -> None:
    tasks = load_official_tasks(ROOT / "data/single_turn_tasks.csv")
    assert len(tasks) == 42
    assert len({task.task_id for task in tasks}) == 42
    assert len({task.normalized_fingerprint for task in tasks}) == 42
    assert len({task.skill_id for task in tasks}) == 14
    assert {task.difficulty for task in tasks} == {"easy", "medium", "hard"}
    assert all(task.split == "official_test" for task in tasks)
    # The released benchmark uses task-specific rubrics: most have five
    # criteria, but a few have six, seven, or nine. The invariant is a valid
    # positive rubric normalized to 100 points, not a fixed criterion count.
    assert {len(task.rubric) for task in tasks} == {5, 6, 7, 9}
    assert all(abs(sum(item.points for item in task.rubric) - 100) < 1e-6 for task in tasks)

    per_skill: dict[str, int] = {}
    for task in tasks:
        per_skill[task.skill_id] = per_skill.get(task.skill_id, 0) + 1
    assert set(per_skill.values()) == {3}


def test_messages_preserve_released_skill_text() -> None:
    task = load_official_tasks(ROOT / "data/single_turn_tasks.csv")[0]
    skill = load_skill_text(ROOT / "skills/single_turn", task.skill_id)
    no_skill = build_messages(task)
    with_skill = build_messages(task, skill)

    assert no_skill[-1]["content"] == with_skill[-1]["content"]
    assert "<skill>" not in no_skill[0]["content"]
    assert skill.strip() in with_skill[0]["content"]
