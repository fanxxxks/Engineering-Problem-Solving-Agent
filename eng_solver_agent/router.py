"""Rule-first subject router for engineering questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eng_solver_agent.debug_logger import log_route, step


@dataclass(frozen=True)
class RouteDecision:
    subject: str
    confidence: float
    matched_rules: tuple[str, ...]


class QuestionRouter:
    """Maps a question to one of the supported subjects.

    Enhanced for Chinese competition questions with matrix literals,
    determinant notation, and equation systems.
    """

    _RULES: dict[str, tuple[str, ...]] = {
        "physics": (
            "physics", "force", "velocity", "acceleration", "mass",
            "energy", "momentum", "newton", "friction", "gravity",
            "力", "速度", "加速度", "质量", "能量", "动量", "牛顿",
        ),
        "circuits": (
            "circuit", "resistor", "resistance", "voltage", "current",
            "capacitance", "inductor", "impedance", "ohm", "node",
            "电路", "电阻", "电压", "电流", "串联", "并联", "欧姆",
        ),
        "linalg": (
            "matrix", "vector", "eigen", "determinant", "rank",
            "basis", "dimension", "linear algebra", "subspace", "orthogonal",
            "矩阵", "向量", "特征值", "特征向量", "行列式", "秩", "逆",
            "线性", "方程组", "基", "维数", "子空间",
        ),
        "calculus": (
            "derivative", "integral", "limit", "differentiate", "differentiation",
            "integration", "gradient", "series", "tangent", "optimization",
            "导数", "积分", "极限", "微分", "泰勒", "级数",
        ),
    }
    _PRIORITY = ("circuits", "physics", "calculus", "linalg")
    _DEFAULT_SUBJECT = "physics"
    _DEFAULT_CONFIDENCE = 0.35

    # Pre-compiled structural patterns
    _MATRIX_LITERAL_RE = re.compile(r"\[\s*\[.+?\]\s*\]")
    _DETERMINANT_RE = re.compile(r"\|\s*[^|]+?[;；]\s*[^|]+?\s*\|")
    _EQUATION_SYSTEM_RE = re.compile(r"^[\s]*[a-zA-Zαβγδ]+[\d]*\s*[+\-]?\s*[a-zA-Zαβγδ]+[\d]*\s*[+\-]?\s*[a-zA-Zαβγδ]*[\d]*\s*=\s*\d", re.MULTILINE)
    _POWER_NOTATION_RE = re.compile(r"\^\{\d+\}|\^\d+")

    def route(self, question: dict[str, Any]) -> str:
        return self.route_with_confidence(question).subject

    def route_with_confidence(self, question: dict[str, Any]) -> RouteDecision:
        text = self._extract_text(question)
        scores: dict[str, int] = {}
        matches: dict[str, list[str]] = {}

        for subject, keywords in self._RULES.items():
            score = 0
            matched: list[str] = []
            for keyword in keywords:
                if keyword in text:
                    score += 1
                    matched.append(keyword)
            scores[subject] = score
            matches[subject] = matched

        # Structural boosts for matrix literals, determinants, equation systems
        self._apply_structural_boosts(text, scores, matches)

        best_subject = max(
            self._RULES,
            key=lambda subject: (scores[subject], -self._priority_index(subject)),
        )
        best_score = scores[best_subject]
        if best_score == 0:
            return RouteDecision(subject=self._DEFAULT_SUBJECT, confidence=self._DEFAULT_CONFIDENCE, matched_rules=())

        total_matches = sum(scores.values()) or 1
        confidence = min(0.99, 0.55 + (best_score / total_matches) * 0.35)
        if self._is_conflicted(scores):
            confidence = max(confidence, 0.72)
        decision = RouteDecision(
            subject=best_subject,
            confidence=round(confidence, 2),
            matched_rules=tuple(matches[best_subject]),
        )
        log_route(decision.subject, decision.confidence, scores, decision.matched_rules)
        return decision

    def _apply_structural_boosts(self, text: str, scores: dict[str, int], matches: dict[str, list[str]]) -> None:
        """Boost scores based on structural patterns (matrix literals, etc.)."""
        # Matrix literal [[1,2],[3,4]] → linalg
        if self._MATRIX_LITERAL_RE.search(text):
            scores["linalg"] = scores.get("linalg", 0) + 3
            matches.setdefault("linalg", []).append("[[matrix_literal]]")
        # Determinant notation |1 2; 3 4| → linalg
        if self._DETERMINANT_RE.search(text):
            scores["linalg"] = scores.get("linalg", 0) + 3
            matches.setdefault("linalg", []).append("|det_notation|")
        # Equation system with multiple lines → linalg
        if self._EQUATION_SYSTEM_RE.search(text):
            scores["linalg"] = scores.get("linalg", 0) + 2
            matches.setdefault("linalg", []).append("equation_system")
        # Matrix power notation A^{2025} → linalg
        if self._POWER_NOTATION_RE.search(text) and ("矩阵" in text or "matrix" in text or "a =" in text.lower() or "设 a" in text):
            scores["linalg"] = scores.get("linalg", 0) + 2
            matches.setdefault("linalg", []).append("matrix_power")
        # Explicit subject field from question dict
        explicit_subject = self._extract_explicit_subject(text)
        if explicit_subject:
            scores[explicit_subject] = scores.get(explicit_subject, 0) + 5

    def _extract_explicit_subject(self, text: str) -> str | None:
        """Check if text contains explicit subject indicators.
        Only apply boost for strong, subject-specific signals."""
        # Strong linear algebra signals (not generic "matrix")
        if any(k in text for k in ("行列式", "det(", "特征值", "特征向量", "eigenvalue", "eigenvector", "线性相关", "线性无关", "秩", "rank of")):
            return "linalg"
        # Strong calculus signals
        if any(k in text for k in ("求导数", "求微分", "求积分", "求极限", "differentiate ", "derivative of", "integral of", "limit of")):
            return "calculus"
        # Strong circuit signals
        if any(k in text for k in ("串联", "并联", "等效电阻", "equivalent resistance", "节点电压", "网孔电流")):
            return "circuits"
        return None

    def _extract_text(self, question: dict[str, Any]) -> str:
        parts = [
            question.get("question"),
            question.get("prompt"),
            question.get("topic"),
            question.get("type"),
            question.get("subject"),
        ]
        return " ".join(str(part) for part in parts if part).lower()

    def _priority_index(self, subject: str) -> int:
        return self._PRIORITY.index(subject) if subject in self._PRIORITY else len(self._PRIORITY)

    def _is_conflicted(self, scores: dict[str, int]) -> bool:
        non_zero = sorted(score for score in scores.values() if score > 0)
        return len(non_zero) >= 2 and non_zero[-1] == non_zero[-2]
