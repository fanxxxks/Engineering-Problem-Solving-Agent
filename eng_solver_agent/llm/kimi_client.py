"""Kimi chat client backed by the OpenAI Python SDK.

Supports multimodal messages (text + image), thinking/reasoning extraction,
structured JSON output, and robust stream recovery from brotli/429 errors.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
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
                request_timeout_seconds or os.getenv("REQUEST_TIMEOUT_SECONDS", "300")
            ),
            max_retry=_to_int(max_retry or os.getenv("MAX_RETRY", "1")),
            temperature=_to_float(temperature or os.getenv("KIMI_TEMPERATURE", "1.0")),
        )
        # Ensure base_url includes /v1 (OpenAI SDK appends /chat/completions)
        base = self.config.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"

        import httpx

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=base,
            timeout=httpx.Timeout(
                connect=30.0,
                read=self.config.request_timeout_seconds,
                write=60.0,
                pool=10.0,
            ),
            max_retries=self.config.max_retry,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, Any]], temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        log_llm_request(messages, model=self.config.model, temperature=temp)

        timeout = self.config.request_timeout_seconds
        collected_content: list[str] = []
        collected_reasoning: list[str] = []

        try:
            stream = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                stream=True,
                stream_options={"include_usage": True},
                timeout=timeout,
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        collected_content.append(delta.content)
                    reasoning = _extract_reasoning_delta(delta)
                    if reasoning:
                        collected_reasoning.append(reasoning)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
        except Exception as exc:
            return self._handle_stream_error(exc, messages, temp, collected_content, collected_reasoning)

        content = "".join(collected_content).strip()
        if collected_reasoning:
            log_llm_thinking("".join(collected_reasoning))

        if not content:
            raise ValueError("empty response content")

        result = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        log_llm_response(result)
        return result

    def _handle_stream_error(
        self,
        exc: Exception,
        messages: list[dict[str, Any]],
        temperature: float,
        collected_content: list[str],
        collected_reasoning: list[str],
    ) -> str:
        """Handle stream errors gracefully: brotli corruption, 429 rate-limit, etc."""
        exc_str = str(exc)
        msg_lower = exc_str.lower()

        # ── Partial content recovery (brotli / connection drop) ──────────
        if collected_content:
            partial = "".join(collected_content).strip()
            if partial:
                log_llm_error(exc, 1, self.config.max_retry + 1)
                print(f"  [WARN] Stream interrupted (brotli/connection), returning partial content ({len(partial)} chars)")
                if collected_reasoning:
                    log_llm_thinking("".join(collected_reasoning))
                result = json.dumps(partial, ensure_ascii=False) if isinstance(partial, (dict, list)) else str(partial)
                log_llm_response(result)
                return result

        # ── 429 rate-limit → retry with exponential backoff + jitter ────
        is_overloaded = ("overloaded" in msg_lower or "429" in exc_str or
                         "rate limit" in msg_lower or "too many requests" in msg_lower)

        max_retries = max(self.config.max_retry, 2)
        for attempt in range(max_retries):
            delay = min(2 ** attempt + random.uniform(0, 2), 60)
            if is_overloaded:
                print(f"  [RATE-LIMIT] Retry {attempt + 1}/{max_retries} after {delay:.1f}s sleep")
                time.sleep(delay)
            else:
                time.sleep(1.0 + random.uniform(0, 1))
            try:
                # Non-streaming fallback for reliability
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                    timeout=self.config.request_timeout_seconds,
                )
                content = response.choices[0].message.content or ""
                reasoning = _extract_reasoning(response.choices[0])
                if reasoning:
                    log_llm_thinking(reasoning)
                if not content.strip():
                    raise ValueError("empty response content after retry")
                result = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
                log_llm_response(result)
                return result
            except Exception as retry_exc:
                exc = retry_exc
                log_llm_error(exc, attempt + 2, max_retries + 1)

        log_llm_error(exc, self.config.max_retry + 1, self.config.max_retry + 1)
        raise exc

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

def _extract_reasoning_delta(delta: Any) -> str | None:
    """Extract reasoning tokens from a streaming delta chunk."""
    candidates = [
        getattr(delta, "reasoning_content", None),
        getattr(delta, "reasoning", None),
        getattr(delta, "thinking", None),
    ]
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "thinking", "cot"):
            val = extra.get(key)
            if val:
                candidates.append(val)
    for c in candidates:
        if c and str(c).strip():
            return str(c)
    return None


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
