"""Centralized constants to eliminate magic numbers across the codebase."""

from __future__ import annotations

# ------------------------------------------------------------------------------
# ReAct reasoning limits
# ------------------------------------------------------------------------------
MAX_REACT_STEPS = 8

# ------------------------------------------------------------------------------
# Concurrency limits
# ------------------------------------------------------------------------------
DEFAULT_MAX_CONCURRENT = 5

# ------------------------------------------------------------------------------
# Numeric tolerances
# ------------------------------------------------------------------------------
NUMERIC_TOLERANCE = 1e-6

# ------------------------------------------------------------------------------
# Embedding / retrieval
# ------------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FAISS_INDEX_DIR = "faiss_index"
DEFAULT_RETRIEVAL_TOP_K = 3
RETRIEVAL_CANDIDATE_MULTIPLIER = 4  # Retrieve N*top_k candidates before filtering

# ------------------------------------------------------------------------------
# Router priorities (higher index = lower priority)
# ------------------------------------------------------------------------------
SUBJECT_PRIORITY = ("circuits", "physics", "calculus", "linalg")
DEFAULT_SUBJECT = "physics"
DEFAULT_CONFIDENCE = 0.35

# ------------------------------------------------------------------------------
# HTTP / LLM client
# ------------------------------------------------------------------------------
DEFAULT_LLM_TIMEOUT_SECONDS = 120
DEFAULT_LLM_MAX_TOKENS = 4096
DEFAULT_LLM_TEMPERATURE = 0.1

# ------------------------------------------------------------------------------
# Exec engine sandbox
# ------------------------------------------------------------------------------
EXEC_TIMEOUT_SECONDS = 5

# ------------------------------------------------------------------------------
# Supported subjects
# ------------------------------------------------------------------------------
SUPPORTED_SUBJECTS = frozenset({"physics", "circuits", "linalg", "calculus"})
