"""Kimi chat client backed by the OpenAI Python SDK.

Supports multimodal messages (text + image), thinking/reasoning extraction,
and structured JSON output.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

from eng_solver_agent.debug_logger import (
    is_verbose,
    log_llm_request,
    log_llm_response,
    log_llm_error,
    log_json_parse,
    log_llm_thinking,
)


@dataclass(frozen=True)
class KimiConfig:
    api_key: str
    base_url: str
    model: str
    request_timeout_seconds: float
    max_retry: int
    temperature: float


class KimiClient:
    """OpenAI-compatible chat client targeting Kimi / Moonshot API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        endpoint_path: str | None = None,
        request_timeout_seconds: float | None = None,
        max_retry: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.config = KimiConfig(
            api_key=api_key or os.getenv("KIMI_API_KEY", ""),
            base_url=base_url or os.getenv("KIMI_BASE_URL", ""),
            model=model or os.getenv("KIMI_MODEL", "kimi"),
            request_timeout_seconds=_to_float(
                request_timeout_seconds or os.getenv("REQUEST_TIMEOUT_SECONDS", "30")
            ),
            max_retry=_to_int(max_retry or os.getenv("MAX_RETRY", "1")),
            temperature=_to_float(temperature or os.getenv("KIMI_TEMPERATURE", "1.0")),
        )
        # Ensure base_url includes /v1 (OpenAI SDK appends /chat/completions)
        base = self.config.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=base,
            timeout=self.config.request_timeout_seconds,
            max_retries=self.config.max_retry,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, Any]], temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        log_llm_request(messages, model=self.config.model, temperature=temp)

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
            )
        except Exception as exc:
            log_llm_error(exc, 1, self.config.max_retry + 1)
            raise

        choice = response.choices[0]
        content = choice.message.content

        # Extract and log thinking/reasoning if present
        reasoning = _extract_reasoning(choice)
        if reasoning:
            log_llm_thinking(reasoning)

        if not content or not str(content).strip():
            raise ValueError("empty response content")

        result = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        log_llm_response(result)
        return result

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        required_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        content = self.chat(messages, temperature=temperature)

        cleaned = self._extract_json_from_markdown(content)
        try:
            data = json.loads(cleaned)
            log_json_parse(cleaned, success=True)
        except json.JSONDecodeError as exc:
            log_json_parse(content, success=False, error=str(exc))
            raise ValueError("model response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("model response JSON must be an object")
        missing = [key for key in required_keys or () if key not in data]
        if missing:
            raise ValueError(f"model response missing required keys: {missing}")
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_json_from_markdown(self, content: str) -> str:
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`(.*?)`',
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                try:
                    json.loads(extracted)
                    return extracted
                except json.JSONDecodeError:
                    continue
        return content.strip()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_reasoning(choice: Any) -> str | None:
    """Extract model thinking/reasoning from a chat completion choice.

    Tries several known field names used by different providers.
    """
    msg = choice.message
    candidates = [
        getattr(msg, "reasoning_content", None),
        getattr(msg, "reasoning", None),
        getattr(msg, "thinking", None),
    ]
    # Also try model_extra for non-standard fields
    extra = getattr(msg, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "thinking", "cot"):
            val = extra.get(key)
            if val:
                candidates.append(val)

    for c in candidates:
        if c and str(c).strip():
            return str(c).strip()
    return None


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"invalid float value: {value!r}") from exc
