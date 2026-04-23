"""Rule-first subject router for engineering questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteDecision:
    subject: str
    confidence: float
    matched_rules: tuple[str, ...]


class QuestionRouter:
    """Maps a question to one of the supported subjects.

    The router uses keyword rules first, then falls back to a priority-based
    decision when multiple subjects score similarly.
    """

    _RULES: dict[str, tuple[str, ...]] = {
        "physics": (
            "physics",
            "force",
            "velocity",
            "acceleration",
            "mass",
            "energy",
            "momentum",
            "newton",
            "friction",
            "gravity",
        ),
        "circuits": (
            "circuit",
            "resistor",
            "resistance",
            "voltage",
            "current",
            "capacitance",
            "inductor",
            "impedance",
            "ohm",
            "node",
            "电路",
            "电阻",
            "电压",
            "电流",
            "电感",
            "电容",
            "阻尼",
            "rlc",
            "cir_",
            "circuits-",
        ),
        "linalg": (
            "matrix",
            "vector",
            "eigen",
            "determinant",
            "rank",
            "basis",
            "dimension",
            "linear algebra",
            "subspace",
            "orthogonal",
        ),
        "calculus": (
            "derivative",
            "integral",
            "limit",
            "differentiate",
            "differentiation",
            "integration",
            "gradient",
            "series",
            "tangent",
            "optimization",
            "导数",
            "积分",
            "极限",
            "微分",
            "级数",
            "泰勒",
            "\\int",
            "\\iint",
            "\\lim",
            "\\sum",
            "\\partial",
            "beta",
            "gamma",
            "cal_",
            "calculus-",
        ),
    }
    _PRIORITY = ("circuits", "physics", "calculus", "linalg")

    def route(self, question: dict[str, Any]) -> str:
        return self.route_with_confidence(question).subject

    def route_with_confidence(self, question: dict[str, Any]) -> RouteDecision:
        text = self._extract_text(question)
        scores = {subject: 0 for subject in self._RULES}
        matches: dict[str, list[str]] = {subject: [] for subject in self._RULES}

        for subject, keywords in self._RULES.items():
            for keyword in keywords:
                if keyword in text:
                    scores[subject] += 1
                    matches[subject].append(keyword)

        best_subject = max(
            self._RULES,
            key=lambda subject: (scores[subject], -self._priority_index(subject)),
        )
        best_score = scores[best_subject]
        if best_score == 0:
            return RouteDecision(subject="physics", confidence=0.35, matched_rules=())

        total_matches = sum(scores.values()) or 1
        confidence = min(0.99, 0.55 + (best_score / total_matches) * 0.35)
        if self._is_conflicted(scores):
            confidence = max(confidence, 0.72)
        return RouteDecision(
            subject=best_subject,
            confidence=round(confidence, 2),
            matched_rules=tuple(matches[best_subject]),
        )

    def _extract_text(self, question: dict[str, Any]) -> str:
        parts = [
            question.get("question"),
            question.get("prompt"),
            question.get("topic"),
            question.get("type"),
            question.get("subject"),
            question.get("question_id"),
        ]
        return " ".join(str(part) for part in parts if part).lower()

    def _priority_index(self, subject: str) -> int:
        return self._PRIORITY.index(subject) if subject in self._PRIORITY else len(self._PRIORITY)

    def _is_conflicted(self, scores: dict[str, int]) -> bool:
        non_zero = sorted(score for score in scores.values() if score > 0)
        return len(non_zero) >= 2 and non_zero[-1] == non_zero[-2]
