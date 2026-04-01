"""Configuration objects for local and competition execution."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    kimi_api_key: str = ""
    default_route: str = "general"
    retrieval_enabled: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            kimi_api_key=os.getenv("KIMI_API_KEY", ""),
            default_route=os.getenv("ENG_SOLVER_DEFAULT_ROUTE", "general"),
            retrieval_enabled=os.getenv("ENG_SOLVER_RETRIEVAL_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
        )
