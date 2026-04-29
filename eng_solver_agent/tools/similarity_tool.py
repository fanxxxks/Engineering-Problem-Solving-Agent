"""Similar problem finder tool.

Provides search, comparison, and matching capabilities to find similar
problems/questions from the local solved-examples and formula-cards
knowledge bases. This version integrates with the LangChain vector
retriever for semantic similarity while keeping the detailed comparison
and scoring features.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from eng_solver_agent.retrieval.kb_loader import KnowledgeBaseLoader
from eng_solver_agent.retrieval.retriever import Retriever


class SimilarProblemTool:
    """Tool for finding, comparing, and matching similar problems.

    Loads solved examples and formula cards from the local knowledge base
    and provides ranked similarity search with detailed comparison metrics.
    When available, uses the LangChain vector retriever for semantic search;
    otherwise falls back to the legacy keyword retriever.
    """

    def __init__(
        self,
        examples_path: str | Path | None = None,
        formula_cards_path: str | Path | None = None,
        use_vector_search: bool = True,
    ) -> None:
        self.loader = KnowledgeBaseLoader()
        base_dir = Path(__file__).resolve().parent.parent / "retrieval"
        self.examples_path = Path(examples_path) if examples_path else base_dir / "solved_examples.jsonl"
        self.formula_cards_path = Path(formula_cards_path) if formula_cards_path else base_dir / "formula_cards.json"
        self.examples = self._load_examples()
        self.formula_cards = self._load_formula_cards()

        # Try LangChain vector retriever first
        self._vector_retriever: Any | None = None
        if use_vector_search:
            try:
                from eng_solver_agent.retrieval.langchain_retriever import LangChainRetriever

                self._vector_retriever = LangChainRetriever(
                    formula_cards=self.formula_cards,
                    solved_examples=self.examples,
                )
            except Exception:
                pass

        # Always keep a legacy keyword retriever as fallback
        self._keyword_retriever = Retriever(
            formula_cards=self.formula_cards,
            solved_examples=self.examples,
        )

        # Try to load embedding model for vector-based comparison
        self._embeddings: Any | None = None
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="./local_models/sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception:
            pass

        # Try to load Cross-Encoder for fine re-rank (Stage 2)
        self._reranker: Any | None = None
        try:
            from eng_solver_agent.retrieval.reranker import BGEReranker
            self._reranker = BGEReranker()
            self._reranker.load()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, query: str) -> str:
        """Entry-point compatible with the generic tool interface.

        Args:
            query: A JSON string or natural-language query.
                JSON format: '{"question": "...", "subject": "...", "top_k": 3}'
        """
        try:
            parsed = json.loads(query)
            if isinstance(parsed, dict):
                question = parsed.get("question", query)
                subject = parsed.get("subject")
                topic = parsed.get("topic")
                top_k = parsed.get("top_k", 3)
                return json.dumps(
                    self.find_similar(question, subject=subject, topic=topic, top_k=top_k),
                    ensure_ascii=False,
                )
        except Exception:
            pass
        return json.dumps(self.find_similar(query), ensure_ascii=False)

    def find_similar(
        self,
        query: str,
        subject: str | None = None,
        topic: str | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Search for similar problems and return ranked results with scores.

        Args:
            query: The question text to search against.
            subject: Optional subject filter (physics, circuits, linalg, calculus).
            topic: Optional topic filter.
            top_k: Maximum number of results to return.

        Returns:
            A dict containing matched_examples, matched_formulas, and metadata.
        """
        if not query or not isinstance(query, str):
            return {"matched_examples": [], "matched_formulas": [], "metadata": {"error": "empty query"}}

        # Use vector retriever if available; otherwise keyword fallback
        # Request top_k * 2 so we have enough after filtering
        recall_k = top_k * 2
        if self._vector_retriever is not None:
            retrieval_result = self._vector_retriever.retrieve(
                query, subject=subject, topic=topic, top_k=recall_k
            )
        else:
            retrieval_result = self._keyword_retriever.retrieve(
                query, subject=subject, topic=topic, top_k=recall_k
            )

        # ── Stage 2: Cross-Encoder re-rank (or fallback to match_score) ──
        if self._reranker is not None and self._reranker.is_available:
            top_examples = self._reranker.rerank(
                query,
                retrieval_result.solved_examples,
                text_fn=lambda c: f"{c.get('question','')} {c.get('reasoning_process','')[:300]} {c.get('answer','')}",
                top_k=top_k,
            )
            top_formulas = self._reranker.rerank(
                query,
                retrieval_result.formula_cards,
                text_fn=lambda c: f"{c.get('topic','')} {c.get('formula','')} {' '.join(c.get('conditions',[]))}",
                top_k=top_k,
            )
        else:
            # Fallback: embedding-cosine match_score
            scored_examples = []
            for example in retrieval_result.solved_examples:
                score_detail = self.match_score(query, example)
                if score_detail["overall_score"] > 0:
                    scored_examples.append((score_detail["overall_score"], score_detail, example))
            scored_examples.sort(key=lambda x: (-x[0], x[2].get("question_id", "")))
            top_examples_raw = scored_examples[:top_k]

            scored_formulas = []
            for card in retrieval_result.formula_cards:
                score_detail = self._match_formula_score(query, card)
                if score_detail["overall_score"] > 0:
                    scored_formulas.append((score_detail["overall_score"], score_detail, card))
            scored_formulas.sort(key=lambda x: (-x[0], x[2].get("id", "")))
            top_formulas_raw = scored_formulas[:top_k]

            # Convert to same format as reranker output
            top_examples = [
                {**ex, "relevance_score": score}
                for score, _detail, ex in top_examples_raw
            ]
            top_formulas = [
                {**card, "relevance_score": score}
                for score, _detail, card in top_formulas_raw
            ]

        return {
            "query": query,
            "subject": subject,
            "topic": topic,
            "matched_examples": [
                {
                    "question_id": ex.get("question_id"),
                    "subject": ex.get("subject"),
                    "topic": ex.get("topic"),
                    "question": ex.get("question"),
                    "answer": ex.get("answer"),
                    "reasoning_process": ex.get("reasoning_process", "")[:200],
                    "similarity_score": ex.get("relevance_score", 0.0),
                }
                for ex in top_examples
            ],
            "matched_formulas": [
                {
                    "id": card.get("id"),
                    "subject": card.get("subject"),
                    "topic": card.get("topic"),
                    "formula": card.get("formula"),
                    "conditions": card.get("conditions", [])[:3],
                    "similarity_score": card.get("relevance_score", 0.0),
                }
                for card in top_formulas
            ],
            "metadata": {
                "total_examples_searched": len(self.examples),
                "total_formulas_searched": len(self.formula_cards),
                "matched_terms": retrieval_result.matched_terms,
                "vector_search_enabled": self._vector_retriever is not None,
            },
        }

    def _cosine(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        import numpy as np
        va = np.array(a, dtype=np.float64)
        vb = np.array(b, dtype=np.float64)
        dot = float(np.dot(va, vb))
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def compare_questions(self, question_a: str, question_b: str) -> dict[str, Any]:
        """Compare two questions using semantic embedding similarity.

        When HuggingFace embeddings are available, uses cosine similarity
        between dense vectors (weight 70%) combined with structural feature
        overlap (weight 30%). Falls back to Jaccard + keyword rules otherwise.
        """
        # ── Vector-based path ──────────────────────────────────────────
        if self._embeddings is not None:
            vec_a = self._embeddings.embed_query(self._normalize_text(question_a))
            vec_b = self._embeddings.embed_query(self._normalize_text(question_b))
            semantic_sim = self._cosine(vec_a, vec_b)

            struct_a = self._extract_structural_features(question_a)
            struct_b = self._extract_structural_features(question_b)
            overlap = sum(1 for k, v in struct_a.items() if struct_b.get(k) == v)
            total = max(len(struct_a), len(struct_b), 1)
            struct_sim = overlap / total

            overall = semantic_sim * 0.7 + struct_sim * 0.3
            return {
                "method": "vector",
                "semantic_similarity": round(semantic_sim, 4),
                "structural_similarity": round(struct_sim, 4),
                "overall_similarity": round(overall, 4),
            }

        # ── Rule-based fallback ────────────────────────────────────────
        tokens_a = set(self._tokenize(self._normalize_text(question_a)))
        tokens_b = set(self._tokenize(self._normalize_text(question_b)))

        if not tokens_a or not tokens_b:
            return {
                "method": "rule",
                "semantic_similarity": 0.0,
                "structural_similarity": 0.0,
                "overall_similarity": 0.0,
            }

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union) if union else 0.0

        struct_a = self._extract_structural_features(question_a)
        struct_b = self._extract_structural_features(question_b)
        struct_overlap = sum(1 for k, v in struct_a.items() if struct_b.get(k) == v)
        struct_total = max(len(struct_a), len(struct_b), 1)
        struct_sim = struct_overlap / struct_total

        math_keywords = {
            "导数", "积分", "极限", "行列式", "矩阵", "特征值", "特征向量",
            "电阻", "电路", "串联", "并联", "牛顿", "动量", "能量", "功",
            "derivative", "integral", "limit", "determinant", "matrix",
            "eigenvalue", "eigenvector", "resistor", "circuit", "series",
            "parallel", "newton", "momentum", "energy", "work",
        }
        keywords_a = tokens_a & math_keywords
        keywords_b = tokens_b & math_keywords
        kw_overlap = len(keywords_a & keywords_b)
        kw_total = max(len(keywords_a), len(keywords_b), 1)
        keyword_sim = kw_overlap / kw_total

        overall = jaccard * 0.4 + struct_sim * 0.35 + keyword_sim * 0.25
        return {
            "method": "rule",
            "semantic_similarity": round(jaccard * 0.6 + keyword_sim * 0.4, 4),
            "structural_similarity": round(struct_sim, 4),
            "overall_similarity": round(overall, 4),
        }

    def match_score(self, query: str, candidate: dict[str, Any]) -> dict[str, Any]:
        """Calculate a detailed match score using semantic embeddings.

        Uses cosine similarity between dense embedding vectors (from HuggingFace
        all-MiniLM-L6-v2) combined with structural feature matching, plus
        subject/topic boosts. Falls back to Jaccard + keyword rules when
        embeddings are unavailable.
        """
        candidate_text = " ".join(
            str(candidate.get(field, "")) for field in ("question", "reasoning_process", "answer", "topic")
        )
        comparison = self.compare_questions(query, candidate_text)

        # Subject match boost
        subject_boost = 0.0
        query_lower = query.lower()
        candidate_subject = str(candidate.get("subject", "")).lower()
        if candidate_subject and candidate_subject in query_lower:
            subject_boost = 0.15

        # Topic match boost
        topic_boost = 0.0
        candidate_topic = str(candidate.get("topic", "")).lower()
        if candidate_topic and candidate_topic in query_lower:
            topic_boost = 0.1

        overall = min(1.0, comparison["overall_similarity"] + subject_boost + topic_boost)

        return {
            "method": comparison.get("method", "rule"),
            "semantic_similarity": comparison["semantic_similarity"],
            "structural_similarity": comparison["structural_similarity"],
            "subject_boost": round(subject_boost, 4),
            "topic_boost": round(topic_boost, 4),
            "overall_score": round(overall, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_examples(self) -> list[dict[str, Any]]:
        if not self.examples_path.exists():
            return []
        try:
            return self.loader.load(self.examples_path)
        except Exception:
            return []

    def _load_formula_cards(self) -> list[dict[str, Any]]:
        if not self.formula_cards_path.exists():
            return []
        try:
            return self.loader.load(self.formula_cards_path)
        except Exception:
            return []

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace("，", ",").replace("。", ".").replace("；", ";")
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())

    def _extract_structural_features(self, text: str) -> dict[str, Any]:
        """Extract structural features from question text for comparison."""
        features: dict[str, Any] = {}

        # Matrix patterns
        if re.search(r"\[\s*\[.+?\]\s*\]", text):
            features["has_matrix_literal"] = True
        if re.search(r"\|\s*[^|]+?\s*\|", text):
            features["has_determinant_notation"] = True

        # Numeric lists (resistors, etc.)
        if re.findall(r"\d+\s*[Ω欧姆]", text):
            features["has_resistor_values"] = True

        # Equation patterns
        if "=" in text:
            features["has_equation"] = True

        # Calculus patterns
        if any(k in text for k in ("∫", "∂", "d/d", "lim", "→")):
            features["has_calculus_symbol"] = True

        # Physics patterns
        if any(k in text for k in ("kg", "m/s", "N", "J", "W", "Ω", "V", "A")):
            features["has_physics_units"] = True

        # Question type
        if any(k in text for k in ("证明", "prove", "求证")):
            features["question_type"] = "proof"
        elif any(k in text for k in ("计算", "求", "compute", "calculate", "find")):
            features["question_type"] = "compute"
        else:
            features["question_type"] = "unknown"

        return features

    def _match_formula_score(self, query: str, card: dict[str, Any]) -> dict[str, Any]:
        """Score a formula card against a query using semantic similarity."""
        card_text = " ".join(
            str(card.get(field, "")) for field in ("topic", "formula", "conditions", "common_traps")
        )
        comparison = self.compare_questions(query, card_text)

        subject_boost = 0.0
        candidate_subject = str(card.get("subject", "")).lower()
        if candidate_subject and candidate_subject in query.lower():
            subject_boost = 0.15

        topic_boost = 0.0
        candidate_topic = str(card.get("topic", "")).lower()
        if candidate_topic and candidate_topic in query.lower():
            topic_boost = 0.1

        overall = min(1.0, comparison["overall_similarity"] + subject_boost + topic_boost)

        return {
            "method": comparison.get("method", "rule"),
            "semantic_similarity": comparison["semantic_similarity"],
            "structural_similarity": comparison["structural_similarity"],
            "subject_boost": round(subject_boost, 4),
            "topic_boost": round(topic_boost, 4),
            "overall_score": round(overall, 4),
        }
