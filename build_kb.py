"""Build the FAISS vector index from local knowledge-base files.

Usage:
    python build_kb.py

This follows the pattern shown in example.py for constructing a
persistent FAISS vector store from structured knowledge data.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    retrieval_dir = Path(__file__).resolve().parent / "eng_solver_agent" / "retrieval"
    formula_path = retrieval_dir / "formula_cards.json"
    examples_path = retrieval_dir / "solved_examples.jsonl"
    index_dir = Path(__file__).resolve().parent / "faiss_index"

    print("=" * 60)
    print("Building FAISS vector index for knowledge base")
    print("=" * 60)

    try:
        from eng_solver_agent.retrieval.langchain_retriever import LangChainRetriever
    except Exception as exc:
        print(f"[ERROR] Failed to import LangChainRetriever: {exc}")
        return 1

    retriever = LangChainRetriever(
        formula_cards_path=formula_path,
        solved_examples_path=examples_path,
        index_dir=index_dir,
    )

    if retriever._vectorstore is None:
        print("[WARN] Vector store initialization failed; check LangChain dependencies.")
        return 1

    try:
        retriever.save_index()
        print(f"[SUCCESS] FAISS index saved to: {index_dir}")
    except Exception as exc:
        print(f"[ERROR] Failed to save index: {exc}")
        return 1

    # Quick smoke test
    result = retriever.retrieve("derivative of x^2", subject="calculus", top_k=2)
    print(f"[TEST] Retrieved {len(result.formula_cards)} formula(s) and {len(result.solved_examples)} example(s)")
    for ex in result.solved_examples:
        print(f"       - {ex.get('question_id')}: {ex.get('question', '')[:50]}...")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
