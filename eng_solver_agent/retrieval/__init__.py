"""Retrieval helpers namespace."""

from eng_solver_agent.retrieval.kb_loader import KnowledgeBaseLoader
from eng_solver_agent.retrieval.retriever import Retriever

# Conditional import so the package does not crash when LangChain is unavailable
try:
    from eng_solver_agent.retrieval.langchain_retriever import LangChainRetriever
except Exception:
    LangChainRetriever = None  # type: ignore[misc,assignment]

__all__ = ["KnowledgeBaseLoader", "Retriever", "LangChainRetriever"]
