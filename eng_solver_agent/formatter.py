"""Helpers for competition output formatting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from eng_solver_agent.schemas import FinalAnswer


def format_submission_item(
    question_id: Any,
    reasoning_process: str,
    answer: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = FinalAnswer(
        question_id=question_id,
        reasoning_process=_require_non_empty(reasoning_process, "reasoning_process"),
        answer=_require_non_empty(answer, "answer"),
        **extra,
    )
    data = payload.model_dump()
    return {
        "question_id": data["question_id"],
        "reasoning_process": data["reasoning_process"],
        "answer": data["answer"],
    }


def format_submission_batch(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [format_submission_item(**item) for item in items]


def format_submission_output(payload: dict[str, Any] | Iterable[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, dict):
        return format_submission_item(**payload)
    return format_submission_batch(payload)


def _require_non_empty(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text
