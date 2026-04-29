"""Cross-Encoder re-ranker using BAAI/bge-reranker-v2-m3.

Multi-stage retrieval pipeline:
  Stage 1 — FAISS vector coarse recall → Top-100 candidates
  Stage 2 — Cross-Encoder fine re-rank → Top-5 final results

Supports GPU (CUDA/MPS) acceleration when available; falls back to CPU.
The model is loaded from a local path (downloaded via modelscope) or
from HuggingFace Hub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


# Local paths to try before hitting the network
_LOCAL_MODEL_CANDIDATES = [
    "./local_models/AI-ModelScope/bge-reranker-v2-m3",
    "./local_models/BAAI/bge-reranker-v2-m3",
]

# HuggingFace model id (used only if no local copy found)
_HF_MODEL_ID = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    """Cross-Encoder re-ranker backed by BAAI/bge-reranker-v2-m3.

    Usage::

        reranker = BGEReranker()
        top5 = reranker.rerank(
            query="Find the derivative of x^2",
            candidates=[{"text": "..."}, {"text": "..."}],
            text_fn=lambda c: c["text"],
            top_k=5,
        )
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Args:
            model_name: Path or HF id for the Cross-Encoder model.
                        Defaults to local candidates then BAAI/bge-reranker-v2-m3.
        """
        self._model: Any = None
        self._device: str = "cpu"
        self._model_path: str | None = None

        # Resolve model path
        resolved = model_name or self._resolve_local()
        if resolved is None:
            # No local copy — will try HF Hub (needs network)
            resolved = _HF_MODEL_ID

        self._model_path = resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when the model has been successfully loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> bool:
        """Load the Cross-Encoder model (idempotent)."""
        if self._model is not None:
            return True

        try:
            from sentence_transformers import CrossEncoder

            device = self._detect_device()
            self._model = CrossEncoder(
                self._model_path,
                device=device,
                trust_remote_code=True,
            )
            self._device = device
            return True
        except Exception:
            return False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        text_fn: Callable[[dict[str, Any]], str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank candidates using the Cross-Encoder.

        Args:
            query: The user's question text.
            candidates: List of candidate dicts (e.g. solved examples).
            text_fn: Function that extracts the text to compare from a candidate.
            top_k: Number of top results to return.

        Returns:
            Candidates sorted by relevance_score descending, limited to top_k.
            Each candidate gains a ``relevance_score`` field (float 0–1).
        """
        if not candidates:
            return []

        if self._model is None:
            # Cross-Encoder not available — return first top_k unchanged
            return self._fallback_sort(query, candidates, top_k)

        # Build (query, candidate_text) pairs
        texts = [text_fn(c) for c in candidates]
        pairs = [(query, t) for t in texts]

        # Predict relevance scores
        scores = self._model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        # Normalize to [0, 1] via sigmoid
        scores = _sigmoid(np.array(scores))

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["relevance_score"] = round(float(score), 4)

        ranked = sorted(
            candidates,
            key=lambda c: c.get("relevance_score", 0.0),
            reverse=True,
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fallback_sort(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return top_k candidates unchanged when Cross-Encoder is unavailable."""
        for c in candidates:
            c.setdefault("relevance_score", 0.0)
        return candidates[:top_k]

    @staticmethod
    def _resolve_local() -> str | None:
        for candidate in _LOCAL_MODEL_CANDIDATES:
            path = Path(candidate).resolve()
            if path.is_dir():
                return str(path)
        return None

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))
