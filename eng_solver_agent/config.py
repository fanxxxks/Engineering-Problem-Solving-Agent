"""Configuration objects for local and competition execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """Load environment variables from .env file if exists."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


# Load .env file on module import
_load_dotenv()


@dataclass(slots=True)
class Settings:
    kimi_api_key: str = ""
    default_route: str = "general"
    retrieval_enabled: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        # Ensure .env is loaded
        _load_dotenv()
        return cls(
            kimi_api_key=os.getenv("KIMI_API_KEY", ""),
            default_route=os.getenv("ENG_SOLVER_DEFAULT_ROUTE", "general"),
            retrieval_enabled=os.getenv("ENG_SOLVER_RETRIEVAL_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
        )
