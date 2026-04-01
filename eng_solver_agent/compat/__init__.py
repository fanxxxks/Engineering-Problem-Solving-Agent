"""Compatibility helpers used when optional third-party packages are absent."""

from eng_solver_agent.compat.pydantic_compat import BaseModel, Field, ValidationError

__all__ = ["BaseModel", "Field", "ValidationError"]
