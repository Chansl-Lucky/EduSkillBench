import pytest
from rlvr.judge import (
    JudgeProtocolError,
    JudgeResult,
    TokenPlanJudge,
    _extract_json,
    weighted_rubric_reward,
)

RUBRIC = [
    {"criterion": "A", "points": 60, "description": "first"},
    {"criterion": "B", "points": 40, "description": "second"},
]


def test_extracts_fenced_json() -> None:
    assert _extract_json('prefix ```json {"criteria": []} ``` suffix') == {"criteria": []}


def test_weighted_reward_uses_task_points() -> None:
    result = JudgeResult.model_validate(
        {"criteria": [{"name": "A", "score": 1}, {"name": "B", "score": 0.5}]}
    )
    assert weighted_rubric_reward(result, RUBRIC) == pytest.approx(0.8)


def test_missing_criterion_is_not_silently_zeroed() -> None:
    result = JudgeResult.model_validate({"criteria": [{"name": "A", "score": 1}]})
    with pytest.raises(JudgeProtocolError):
        weighted_rubric_reward(result, RUBRIC)


@pytest.mark.asyncio
async def test_length_failure_expands_budget_before_accepting_reward() -> None:
    judge = TokenPlanJudge(
        api_key="test-key",
        model="test-model",
        max_tokens=16,
        max_tokens_cap=32,
    )
    budgets: list[int] = []

    async def fake_post(payload: dict[str, object]) -> dict[str, object]:
        budget = int(payload["max_tokens"])
        budgets.append(budget)
        if budget == 16:
            return {
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 16},
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"criteria":[{"name":"A","score":1},'
                            '{"name":"B","score":0.5}],"fatal_errors":[],"overall":0.8}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 24},
        }

    judge._post_with_retries = fake_post  # type: ignore[method-assign]
    try:
        result = await judge.score("prompt", "completion", RUBRIC)
    finally:
        await judge.aclose()

    assert budgets == [16, 32]
    assert result.max_tokens_used == 32
    assert result.finish_reason == "stop"
    assert result.usage["completion_tokens"] == 24


@pytest.mark.asyncio
async def test_missing_criterion_requests_protocol_repair() -> None:
    judge = TokenPlanJudge(api_key="test-key", model="test-model")
    calls: list[dict[str, object]] = []

    async def fake_post(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload.copy())
        content = (
            '{"criteria":[{"name":"A","score":1}],"fatal_errors":[],"overall":1}'
            if len(calls) == 1
            else (
                '{"criteria":[{"name":"A","score":1},'
                '{"name":"B","score":0.5}],"fatal_errors":[],"overall":0.8}'
            )
        )
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {},
        }

    judge._post_with_retries = fake_post  # type: ignore[method-assign]
    try:
        result = await judge.score("prompt", "completion", RUBRIC)
    finally:
        await judge.aclose()

    assert len(calls) == 2
    assert "failed validation" in str(calls[1]["messages"])
    assert weighted_rubric_reward(result, RUBRIC) == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_paper_protocol_matches_released_verifier_score_handling() -> None:
    judge = TokenPlanJudge(
        api_key="test-key",
        model="test-model",
        protocol="paper_pass_fail",
    )

    async def fake_post(payload: dict[str, object]) -> dict[str, object]:
        assert "Expected behavior (rubric)" in str(payload["messages"])
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"items":[{"criterion":"first","pass":true},'
                            '{"criterion":"second","pass":false}],'
                            '"score":0.99,"reasoning":"test"}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

    judge._post_with_retries = fake_post  # type: ignore[method-assign]
    try:
        result = await judge.score_reward(
            "prompt", "completion", RUBRIC, expected_output="expected"
        )
    finally:
        await judge.aclose()

    # BenchFlow 0.6.7 uses the model's top-level score directly.
    assert result.rubric_reward == 0.99
    assert result.reward == 0.99
    assert result.result.protocol == "paper_pass_fail"
