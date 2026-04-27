"""Basic validation for submission payloads."""

from __future__ import annotations

from typing import Any


from eng_solver_agent.exceptions import SolverError


class SubmissionValidationError(SolverError):
    """Raised when submission format validation fails."""
    pass


REQUIRED_SUBMISSION_KEYS = ("question_id", "reasoning_process", "answer")


def validate_submission_item(item: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_SUBMISSION_KEYS if key not in item]
    if missing:
        raise SubmissionValidationError(f"Missing submission keys: {missing}")

    if not str(item["reasoning_process"]).strip():
        raise SubmissionValidationError("reasoning_process must not be empty")
    if not str(item["answer"]).strip():
        raise SubmissionValidationError("answer must not be empty")


validate_final_answer = validate_submission_item
