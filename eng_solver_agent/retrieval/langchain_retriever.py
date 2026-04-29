"""LangChain-based vector retrieval system.

This module refactors the legacy keyword-based Retriever into a hybrid
vector+keyword retrieval engine using HuggingFaceEmbeddings + FAISS,
following the patterns demonstrated in example.py.

Key improvements over the legacy system:
- Semantic similarity via dense vector embeddings (HuggingFace all-MiniLM-L6-v2)
- Persistent FAISS index with save_local / load_local
- Hybrid scoring: vector distance + keyword overlap + structured filters
- Compatible with LangChain Runnable interfaces
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from eng_solver_agent.schemas import RetrievalResult


# ------------------------------------------------------------------------------
# Optional LangChain imports with graceful fallback
# ------------------------------------------------------------------------------
try:
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS

    # Prefer the dedicated langchain-huggingface package; fall back to
    # langchain-community (deprecated but still works in 0.3.x).
    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    Document = None  # type: ignore[misc,assignment]


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
_EMBEDDING_MODEL = "./local_models/sentence-transformers/all-MiniLM-L6-v2"
_FAISS_INDEX_DIR = "faiss_index"


class LangChainRetriever:
    """Hybrid vector + structured retrieval engine.

    When LangChain dependencies are available, this uses dense vector
    embeddings (HuggingFace all-MiniLM-L6-v2) backed by FAISS for
    semantic similarity search. When unavailable, it transparently
    falls back to the legacy keyword-based Retriever.
    """

    def __init__(
        self,
        formula_cards: list[dict[str, Any]] | None = None,
        solved_examples: list[dict[str, Any]] | None = None,
        formula_cards_path: str | Path | None = None,
        solved_examples_path: str | Path | None = None,
        index_dir: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir) if index_dir else Path(_FAISS_INDEX_DIR)
        self._formula_cards = self._resolve_data(formula_cards, formula_cards_path)
        self._solved_examples = self._resolve_data(solved_examples, solved_examples_path)

        self._vectorstore: Any | None = None
        self._embeddings: Any | None = None
        self._keyword_retriever: Any | None = None

        if LANGCHAIN_AVAILABLE:
            self._init_vectorstore()
        else:
            self._init_fallback()

    # ------------------------------------------------------------------
    # Public API (mirrors legacy Retriever)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        subject: str | None = None,
        topic: str | None = None,
        top_k: int = 3,
    ) -> RetrievalResult:
        """Retrieve relevant formula cards and solved examples.

        Uses hybrid scoring when vector search is available; otherwise
        falls back to pure keyword matching.

        Note: ``top_k`` is the number of *final* results requested. Internally
        we recall ``max(top_k * 10, 50)`` candidates for Cross-Encoder re-rank.
        """
        if not query:
            return RetrievalResult(
                subject=subject, topic=topic, query=query, metadata={"error": "empty query"}
            )

        subject_key = self._normalize_subject(subject)
        # Coarse recall: fetch many candidates for downstream re-rank
        recall_k = max(top_k * 10, 50)

        if self._vectorstore is not None:
            formula_hits = self._vector_search(
                query, doc_type="formula_card", subject=subject_key, topic=topic, top_k=recall_k
            )
            example_hits = self._vector_search(
                query, doc_type="solved_example", subject=subject_key, topic=topic, top_k=recall_k
            )
        else:
            formula_hits, example_hits = self._fallback_keyword_search(
                query, subject_key, topic, recall_k
            )

        matched_terms = sorted(set(self._tokenize(query)))
        return RetrievalResult(
            subject=subject_key,
            topic=topic,
            query=query,
            formula_cards=formula_hits,
            solved_examples=example_hits,
            matched_terms=matched_terms,
            metadata={
                "top_k": top_k,
                "formula_count": len(formula_hits),
                "example_count": len(example_hits),
                "vector_search_enabled": self._vectorstore is not None,
            },
        )

    def search(self, query: str, documents: list[str] | None = None) -> list[str]:
        """Simple text search over an optional document list."""
        if documents is None:
            documents = self._all_texts()
        if not query or not documents:
            return []
        query_terms = set(self._tokenize(query))
        scored = []
        for doc in documents:
            score = len(query_terms & set(self._tokenize(doc)))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, doc in scored]

    def summarize(self, result: RetrievalResult, max_items: int = 2) -> dict[str, list[dict[str, Any]]]:
        return {
            "formula_cards": [self._summarize_formula_card(card) for card in result.formula_cards[:max_items]],
            "solved_examples": [self._summarize_example(ex) for ex in result.solved_examples[:max_items]],
        }

    def save_index(self, index_dir: str | Path | None = None) -> None:
        """Persist the FAISS index to disk."""
        if self._vectorstore is None:
            raise RuntimeError("Vector store is not available")
        target = Path(index_dir) if index_dir else self.index_dir
        target.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(target))

    def load_index(self, index_dir: str | Path | None = None) -> bool:
        """Load a previously saved FAISS index."""
        if not LANGCHAIN_AVAILABLE:
            return False
        target = Path(index_dir) if index_dir else self.index_dir
        if not target.exists():
            return False
        try:
            self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
            self._vectorstore = FAISS.load_local(
                str(target), self._embeddings, allow_dangerous_deserialization=True
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # LangChain vector search
    # ------------------------------------------------------------------

    def _init_vectorstore(self) -> None:
        """Build or load the FAISS vector store (GPU-first, CPU fallback)."""
        if self.load_index():
            # Post-load: try to migrate index to GPU
            self._try_gpu_migrate()
            return

        docs = self._build_documents()
        if not docs:
            self._init_fallback()
            return

        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        self._vectorstore = FAISS.from_documents(docs, self._embeddings)
        self._try_gpu_migrate()
        try:
            self.save_index()
        except Exception:
            pass

    def _try_gpu_migrate(self) -> None:
        """Migrate FAISS index to GPU if available."""
        if self._vectorstore is None:
            return
        try:
            import faiss
            if faiss.get_num_gpus() > 0:
                index = self._vectorstore.index
                if not isinstance(index, faiss.GpuIndex):
                    self._vectorstore.index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            pass

    def _build_documents(self) -> list[Any]:
        """Convert formula cards and solved examples into LangChain Documents."""
        docs: list[Any] = []
        for card in self._formula_cards:
            page_content = self._formula_card_to_text(card)
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "doc_type": "formula_card",
                        "id": card.get("id", ""),
                        "subject": card.get("subject", ""),
                        "topic": card.get("topic", ""),
                        "raw_data": json.dumps(card, ensure_ascii=False),
                    },
                )
            )
        for example in self._solved_examples:
            page_content = self._solved_example_to_text(example)
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "doc_type": "solved_example",
                        "id": example.get("question_id", ""),
                        "subject": example.get("subject", ""),
                        "topic": example.get("topic", ""),
                        "raw_data": json.dumps(example, ensure_ascii=False),
                    },
                )
            )
        return docs

    def _vector_search(
        self,
        query: str,
        doc_type: str,
        subject: str | None,
        topic: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search with post-filtering."""
        if self._vectorstore is None:
            return []

        # Retrieve more candidates to allow for filtering
        candidates = self._vectorstore.similarity_search_with_score(query, k=top_k * 4)

        results: list[tuple[float, dict[str, Any]]] = []
        for doc, score in candidates:
            meta = doc.metadata
            if meta.get("doc_type") != doc_type:
                continue
            if subject and self._normalize_subject(meta.get("subject")) != subject:
                continue
            if topic and topic.lower() not in str(meta.get("topic", "")).lower():
                continue
            raw_data = json.loads(meta.get("raw_data", "{}"))
            # Combine vector distance (lower is better) with keyword boost
            hybrid_score = self._hybrid_score(query, raw_data, score)
            results.append((hybrid_score, raw_data))

        # Sort by hybrid score descending, deduplicate by id
        results.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for _, item in results:
            item_id = item.get("id") or item.get("question_id")
            if item_id and item_id in seen:
                continue
            if item_id:
                seen.add(item_id)
            deduped.append(item)
            if len(deduped) >= top_k:
                break
        return deduped

    def _hybrid_score(self, query: str, item: dict[str, Any], vector_distance: float) -> float:
        """Combine vector distance with keyword overlap into a unified score.

        vector_distance from FAISS L2 (lower = more similar) is inverted so
        that higher values mean better matches, then blended with a keyword
        overlap bonus.
        """
        # Invert L2 distance: use 1 / (1 + dist) to get [0, 1]
        vector_sim = 1.0 / (1.0 + float(vector_distance))

        query_terms = set(self._tokenize(query))
        item_text = " ".join(str(v) for v in item.values() if isinstance(v, (str, int, float)))
        item_terms = set(self._tokenize(item_text))
        keyword_overlap = len(query_terms & item_terms) / max(len(query_terms), 1)

        # Weighted hybrid
        return vector_sim * 0.7 + keyword_overlap * 0.3

    # ------------------------------------------------------------------
    # Fallback keyword search (legacy behaviour)
    # ------------------------------------------------------------------

    def _init_fallback(self) -> None:
        """Initialize the legacy keyword retriever as a fallback."""
        from eng_solver_agent.retrieval.retriever import Retriever as LegacyRetriever

        self._keyword_retriever = LegacyRetriever(
            formula_cards=self._formula_cards,
            solved_examples=self._solved_examples,
        )

    def _fallback_keyword_search(
        self,
        query: str,
        subject: str | None,
        topic: str | None,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self._keyword_retriever is None:
            return [], []
        result = self._keyword_retriever.retrieve(query, subject=subject, topic=topic, top_k=top_k)
        return result.formula_cards, result.solved_examples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_data(
        self,
        explicit: list[dict[str, Any]] | None,
        path: str | Path | None,
    ) -> list[dict[str, Any]]:
        if explicit is not None:
            return explicit
        if path is None:
            return []
        candidate = Path(path)
        if not candidate.exists():
            return []
        try:
            from eng_solver_agent.retrieval.kb_loader import KnowledgeBaseLoader

            return KnowledgeBaseLoader().load(candidate)
        except Exception:
            return []

    def _formula_card_to_text(self, card: dict[str, Any]) -> str:
        parts = [
            f"Subject: {card.get('subject', '')}",
            f"Topic: {card.get('topic', '')}",
            f"Formula: {card.get('formula', '')}",
            f"Conditions: {', '.join(self._as_list(card.get('conditions')))}",
            f"Common traps: {', '.join(self._as_list(card.get('common_traps')))}",
            f"Keywords: {', '.join(self._as_list(card.get('keywords')))}",
        ]
        return "\n".join(parts)

    def _solved_example_to_text(self, example: dict[str, Any]) -> str:
        parts = [
            f"Subject: {example.get('subject', '')}",
            f"Topic: {example.get('topic', '')}",
            f"Question: {example.get('question', '')}",
            f"Reasoning: {example.get('reasoning_process', '')}",
            f"Answer: {example.get('answer', '')}",
            f"Tags: {', '.join(self._as_list(example.get('tags')))}",
        ]
        return "\n".join(parts)

    def _all_texts(self) -> list[str]:
        texts: list[str] = []
        for card in self._formula_cards:
            texts.append(self._formula_card_to_text(card))
        for example in self._solved_examples:
            texts.append(self._solved_example_to_text(example))
        return texts

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(text or "").lower())

    def _normalize_subject(self, subject: Any) -> str | None:
        if subject is None:
            return None
        normalized = str(subject).strip().lower()
        if normalized in {"physics", "circuits", "linalg", "calculus"}:
            return normalized
        if normalized in {"linear_algebra", "matrix", "algebra"}:
            return "linalg"
        return normalized or None

    def _as_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _summarize_formula_card(self, card: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": card.get("id"),
            "subject": card.get("subject"),
            "topic": card.get("topic"),
            "formula": card.get("formula"),
            "conditions": self._as_list(card.get("conditions"))[:3],
        }

    def _summarize_example(self, example: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": example.get("question_id"),
            "subject": example.get("subject"),
            "topic": example.get("topic"),
            "question": example.get("question"),
            "answer": example.get("answer"),
        }
