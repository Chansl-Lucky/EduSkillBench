"""Validation utilities for independent GRPO train/dev prompt pools."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rlvr.data import PromptRecord


def load_prompt_pool(path: str | Path) -> list[PromptRecord]:
    path = Path(path)
    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                record = PromptRecord.model_validate({**item, "source": str(path)})
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid prompt pool record at line {line_number}") from exc
            if record.split not in {"train", "dev", "judge_calibration"}:
                raise ValueError(
                    f"prompt pool split must be train/dev/judge_calibration, got {record.split}"
                )
            if record.task_id in seen_ids:
                raise ValueError(f"duplicate prompt task_id: {record.task_id}")
            if record.normalized_fingerprint in seen_fingerprints:
                raise ValueError(f"duplicate normalized prompt: {record.task_id}")
            seen_ids.add(record.task_id)
            seen_fingerprints.add(record.normalized_fingerprint)
            records.append(record)
    return records


def assert_disjoint(*pools: list[PromptRecord]) -> None:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for pool in pools:
        for record in pool:
            if record.task_id in ids or record.normalized_fingerprint in fingerprints:
                raise ValueError(f"prompt leakage detected at {record.task_id}")
            ids.add(record.task_id)
            fingerprints.add(record.normalized_fingerprint)


def semantic_overlap_report(
    candidates: list[PromptRecord],
    references: list[PromptRecord],
) -> list[dict[str, str | float]]:
    """Return each candidate's closest reference using character n-gram TF-IDF.

    This is a screening heuristic, not proof of semantic independence. High-scoring
    pairs must be reviewed manually before a generated pool is frozen.
    """
    if not candidates or not references:
        return []
    texts = [item.question for item in candidates + references]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(texts)
    similarities = cosine_similarity(matrix[: len(candidates)], matrix[len(candidates) :])
    report: list[dict[str, str | float]] = []
    for row_index, candidate in enumerate(candidates):
        best_index = int(similarities[row_index].argmax())
        report.append(
            {
                "candidate_task_id": candidate.task_id,
                "reference_task_id": references[best_index].task_id,
                "similarity": float(similarities[row_index, best_index]),
            }
        )
    return report


def assert_semantically_separated(
    candidates: list[PromptRecord],
    references: list[PromptRecord],
    *,
    threshold: float = 0.72,
) -> None:
    for match in semantic_overlap_report(candidates, references):
        if float(match["similarity"]) >= threshold:
            raise ValueError(
                "near-duplicate prompt detected: "
                f"{match['candidate_task_id']} vs {match['reference_task_id']} "
                f"(similarity={float(match['similarity']):.3f})"
            )
