"""Smart router using LLM for subject classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eng_solver_agent.router import QuestionRouter, RouteDecision
from eng_solver_agent.llm.kimi_client import KimiClient


@dataclass(frozen=True)
class SmartRouteDecision(RouteDecision):
    """Extended route decision with LLM confidence."""
    llm_confidence: float = 0.0
    reasoning: str = ""


class SmartRouter(QuestionRouter):
    """Enhanced router with LLM-based classification."""

    def __init__(self, kimi_client: KimiClient | None = None) -> None:
        super().__init__()
        self.kimi_client = kimi_client
        self._rule_router = QuestionRouter()

    def route_with_confidence(self, question: dict[str, Any]) -> SmartRouteDecision:
        """Route using both rule-based and LLM-based approaches."""
        # First try rule-based routing
        rule_decision = self._rule_router.route_with_confidence(question)
        
        # If rule-based is confident, use it
        if rule_decision.confidence >= 0.8:
            return SmartRouteDecision(
                subject=rule_decision.subject,
                confidence=rule_decision.confidence,
                matched_rules=rule_decision.matched_rules,
                llm_confidence=0.0,
                reasoning="Rule-based classification"
            )
        
        # Otherwise, try LLM-based routing
        llm_decision = self._llm_route(question)
        if llm_decision:
            return llm_decision
        
        # Fall back to rule-based
        return SmartRouteDecision(
            subject=rule_decision.subject,
            confidence=rule_decision.confidence,
            matched_rules=rule_decision.matched_rules,
            llm_confidence=0.0,
            reasoning="Fallback to rule-based"
        )

    def _llm_route(self, question: dict[str, Any]) -> SmartRouteDecision | None:
        """Use LLM to classify the subject."""
        if self.kimi_client is None:
            return None
            
        question_text = self._extract_text(question)
        if len(question_text) < 10:
            return None
        
        messages = [
            {
                "role": "system",
                "content": """You are a subject classifier for engineering problems.
Classify the given problem into one of: physics, circuits, linalg (linear algebra), calculus.

Respond in strict JSON format:
{
    "subject": "one of: physics, circuits, linalg, calculus",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""
            },
            {
                "role": "user",
                "content": f"Classify this problem:\n\n{question_text[:500]}"
            }
        ]
        
        try:
            response = self.kimi_client.chat_json(
                messages,
                required_keys=["subject", "confidence"]
            )
            
            subject = response.get("subject", "physics").lower().strip()
            confidence = float(response.get("confidence", 0.5))
            reasoning = response.get("reasoning", "")
            
            # Normalize subject
            subject = self._normalize_subject(subject)
            
            if confidence >= 0.6:
                return SmartRouteDecision(
                    subject=subject,
                    confidence=min(confidence, 0.95),
                    matched_rules=(),
                    llm_confidence=confidence,
                    reasoning=reasoning or "LLM classification"
                )
        except Exception:
            pass
        
        return None

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject name."""
        normalized = subject.strip().lower()
        if normalized in {"physics", "circuits", "linalg", "calculus"}:
            return normalized
        if normalized in {"linear_algebra", "matrix", "algebra", "linear algebra"}:
            return "linalg"
        if normalized in {"circuit", "circuit_analysis", "circuit analysis", "electronics"}:
            return "circuits"
        if normalized in {"calc", "analysis", "mathematical analysis"}:
            return "calculus"
        if normalized in {"mechanics", "dynamics", "kinematics", "thermodynamics", "electromagnetism"}:
            return "physics"
        return "physics"

    def _extract_text(self, question: dict[str, Any]) -> str:
        """Extract text from question."""
        parts = [
            question.get("question"),
            question.get("prompt"),
            question.get("topic"),
            question.get("type"),
            question.get("subject"),
        ]
        return " ".join(str(part) for part in parts if part)
