"""Lightweight keyword-based retrieval over local formula and example cards."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eng_solver_agent.retrieval.kb_loader import KnowledgeBaseLoader
from eng_solver_agent.schemas import RetrievalResult


class Retriever:
    """Simple subject-filtered retrieval without external vector databases."""

    def __init__(
        self,
        loader: KnowledgeBaseLoader | None = None,
        formula_cards: list[dict[str, Any]] | None = None,
        solved_examples: list[dict[str, Any]] | None = None,
        formula_cards_path: str | Path | None = None,
        solved_examples_path: str | Path | None = None,
    ) -> None:
        self.loader = loader or KnowledgeBaseLoader()
        self.formula_cards = formula_cards if formula_cards is not None else self._load_if_present(formula_cards_path)
        self.solved_examples = solved_examples if solved_examples is not None else self._load_if_present(solved_examples_path)

    def search(self, query: str, documents: list[str] | None = None) -> list[str]:
        if not query or not documents:
            return []
        query_terms = set(self._tokenize(query))
        scored = []
        for document in documents:
            doc_terms = set(self._tokenize(document))
            score = len(query_terms & doc_terms)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [document for _, document in scored]

    def retrieve(
        self,
        query: str,
        subject: str | None = None,
        topic: str | None = None,
        top_k: int = 3,
    ) -> RetrievalResult:
        subject_key = self._normalize_subject(subject)
        query_terms = set(self._tokenize(query))
        formula_hits = self._rank_cards(self.formula_cards, query_terms, subject_key, topic, top_k)
        example_hits = self._rank_cards(self.solved_examples, query_terms, subject_key, topic, top_k)
        matched_terms = sorted(query_terms)
        return RetrievalResult(
            subject=subject_key,
            topic=topic,
            query=query,
            formula_cards=formula_hits,
            solved_examples=example_hits,
            matched_terms=matched_terms,
            metadata={"top_k": top_k, "formula_count": len(formula_hits), "example_count": len(example_hits)},
        )

    def summarize(self, result: RetrievalResult, max_items: int = 2) -> dict[str, list[dict[str, Any]]]:
        return {
            "formula_cards": [self._summarize_formula_card(card) for card in result.formula_cards[:max_items]],
            "solved_examples": [self._summarize_example(example) for example in result.solved_examples[:max_items]],
        }

    def _load_if_present(self, path: str | Path | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        candidate = Path(path)
        if not candidate.exists():
            return []
        try:
            return self.loader.load(candidate)
        except ValueError:
            return []

    def _rank_cards(
        self,
        cards: list[dict[str, Any]],
        query_terms: set[str],
        subject: str | None,
        topic: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        topic_terms = set(self._tokenize(topic or ""))
        for card in cards:
            if not isinstance(card, dict):
                continue
            if subject and self._normalize_subject(card.get("subject")) != subject:
                continue
            score = self._score_card(card, query_terms, topic_terms)
            if score > 0:
                scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id") or item[1].get("question_id") or "")))
        return [card for _, card in scored[:top_k]]

    def _score_card(self, card: dict[str, Any], query_terms: set[str], topic_terms: set[str]) -> float:
        card_terms = set(self._tokenize(" ".join(self._card_text_fields(card))))
        keywords = set(self._tokenize(" ".join(self._as_list(card.get("keywords")))))
        score = len(query_terms & card_terms) * 2 + len(query_terms & keywords) * 3
        if topic_terms:
            score += len(topic_terms & card_terms) * 2
            score += len(topic_terms & keywords) * 3
        topic_value = str(card.get("topic", "")).lower()
        if topic_value and any(term in topic_value for term in topic_terms):
            score += 2
        if not query_terms:
            score += 0
        return float(score)

    def _card_text_fields(self, card: dict[str, Any]) -> list[str]:
        fields = [
            str(card.get("topic", "")),
            str(card.get("formula", "")),
            " ".join(self._as_list(card.get("conditions"))),
            " ".join(self._as_list(card.get("common_traps"))),
            str(card.get("question", "")),
            str(card.get("reasoning_process", "")),
            str(card.get("answer", "")),
            " ".join(self._as_list(card.get("tags"))),
        ]
        return [field for field in fields if field]

    def _summarize_formula_card(self, card: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": card.get("id"),
            "subject": card.get("subject"),
            "topic": card.get("topic"),
            "formula": card.get("formula"),
            "conditions": self._as_list(card.get("conditions"))[:3],
            "common_traps": self._as_list(card.get("common_traps"))[:3],
        }

    def _summarize_example(self, example: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": example.get("question_id"),
            "subject": example.get("subject"),
            "topic": example.get("topic"),
            "question": example.get("question"),
            "answer": example.get("answer"),
            "tags": self._as_list(example.get("tags"))[:3],
        }

    def _normalize_subject(self, subject: Any) -> str | None:
        if subject is None:
            return None
        normalized = str(subject).strip().lower()
        if normalized in {"physics", "circuits", "linalg", "calculus"}:
            return normalized
        if normalized in {"linear_algebra", "matrix", "algebra"}:
            return "linalg"
        return normalized or None

    def _tokenize(self, text: Any) -> list[str]:
        raw = str(text or "").lower().replace("^", " ")
        return [token for token in re.findall(r"[a-z0-9]+", raw) if token]

    def _as_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        return [str(value)]
