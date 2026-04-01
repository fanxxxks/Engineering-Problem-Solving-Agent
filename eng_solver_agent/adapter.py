"""Input adaptation helpers."""

from __future__ import annotations

from typing import Any


class QuestionAdapter:
    """Normalizes raw competition questions into a predictable shape."""

    def normalize(self, question: dict[str, Any]) -> dict[str, Any]:
        question_id = question.get("question_id") or question.get("id") or "unknown"
        prompt = question.get("prompt") or question.get("question") or ""
        normalized = dict(question)
        normalized["question_id"] = question_id
        normalized["prompt"] = str(prompt)
        return normalized
