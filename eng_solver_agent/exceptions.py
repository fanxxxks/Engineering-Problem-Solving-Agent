"""Unified exception hierarchy for the solver agent.

All domain-specific errors inherit from SolverError so callers can catch
a single base class while still accessing granular error types.
"""

from __future__ import annotations


class SolverError(Exception):
    """Base class for all solver-agent errors."""

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


# ------------------------------------------------------------------------------
# Tool errors
# ------------------------------------------------------------------------------

class ToolError(SolverError):
    """Base class for tool execution errors."""
    pass


class ToolUnsupportedError(ToolError):
    """Raised when a tool encounters an unsupported expression or operation."""
    pass


class ToolInputError(ToolError):
    """Raised when tool arguments are malformed or missing required fields."""
    pass


class ToolTimeoutError(ToolError):
    """Raised when a tool (e.g. exec sandbox) exceeds its time budget."""
    pass


# ------------------------------------------------------------------------------
# LLM errors
# ------------------------------------------------------------------------------

class LLMError(SolverError):
    """Base class for LLM client errors."""
    pass


class LLMResponseError(LLMError):
    """Raised when the LLM returns an empty or malformed response."""
    pass


class LLMJSONError(LLMError):
    """Raised when the LLM response cannot be parsed as JSON."""
    pass


# ------------------------------------------------------------------------------
# Retrieval errors
# ------------------------------------------------------------------------------

class RetrievalError(SolverError):
    """Base class for knowledge-base retrieval errors."""
    pass


class IndexNotFoundError(RetrievalError):
    """Raised when a requested FAISS index does not exist on disk."""
    pass


# ------------------------------------------------------------------------------
# Routing errors
# ------------------------------------------------------------------------------

class RoutingError(SolverError):
    """Raised when the question router cannot determine a valid subject."""
    pass
