"""Basic validation for submission payloads."""

from __future__ import annotations

from typing import Any


REQUIRED_SUBMISSION_KEYS = ("question_id", "reasoning_process", "answer")


def validate_submission_item(item: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SUBMISSION_KEYS if key not in item]
    if missing:
        raise ValueError(f"Missing submission keys: {missing}")

    if not str(item["reasoning_process"]).strip():
        raise ValueError("reasoning_process must not be empty")
    if not str(item["answer"]).strip():
        raise ValueError("answer must not be empty")


def validate_final_answer(item: dict[str, Any]) -> None:
    """Alias for final payload validation to keep call sites explicit."""

    validate_submission_item(item)
