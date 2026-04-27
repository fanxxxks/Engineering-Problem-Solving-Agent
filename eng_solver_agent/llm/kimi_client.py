"""Replaceable Kimi chat client built on the standard library.

This module intentionally avoids assuming a single fixed API shape. The endpoint
path, request field names, and response extraction paths are all kept in
constants or constructor arguments so the implementation can be adapted without
rewriting call sites.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urljoin

from eng_solver_agent.debug_logger import (
    is_verbose,
    log_llm_request,
    log_llm_response,
    log_llm_error,
    log_json_parse,
    log_error,
)


DEFAULT_ENDPOINT_PATH = os.getenv("KIMI_ENDPOINT_PATH", "/v1/chat/completions")
REQUEST_FIELD_MAP = {
    "model": "model",
    "messages": "messages",
    "temperature": "temperature",
}
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Engineering-Problem-Solving-Agent/1.0",
}
AUTH_HEADER_NAME = "Authorization"
AUTH_HEADER_PREFIX = "Bearer "
RESPONSE_CONTENT_PATHS = (
    ("choices", 0, "message", "content"),
    ("choices", 0, "text"),
    ("output_text",),
    ("content",)
)


class _RedirectHandler(urllib_request.HTTPRedirectHandler):
    """Handle HTTP 307/308 redirects for POST requests."""
    def http_error_307(self, req, fp, code, msg, headers):
        return self._do_redirect(req, fp, code, msg, headers)
    def http_error_308(self, req, fp, code, msg, headers):
        return self._do_redirect(req, fp, code, msg, headers)
    def _do_redirect(self, req, fp, code, msg, headers):
        new_url = headers.get("Location") or headers.get("location")
        if new_url is None:
            raise urllib_error.HTTPError(req.full_url, code, msg, headers, fp)
        new_request = urllib_request.Request(
            new_url, data=req.data, headers=dict(req.headers), method=req.get_method()
        )
        return self.parent.open(new_request, timeout=getattr(req, 'timeout', None))


def build_opener():
    """Build a URL opener that handles 307/308 redirects."""
    handler = _RedirectHandler()
    return urllib_request.build_opener(handler).open


@dataclass(frozen=True)
class KimiConfig:
    api_key: str
    base_url: str
    model: str
    endpoint_path: str
    request_timeout_seconds: float
    max_retry: int
    temperature: float


class KimiClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        endpoint_path: str | None = None,
        request_timeout_seconds: float | None = None,
        max_retry: int | None = None,
        temperature: float | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = KimiConfig(
            api_key=api_key if api_key is not None else os.getenv("KIMI_API_KEY", ""),
            base_url=base_url if base_url is not None else os.getenv("KIMI_BASE_URL", ""),
            model=model if model is not None else os.getenv("KIMI_MODEL", "kimi"),
            endpoint_path=endpoint_path if endpoint_path is not None else os.getenv("KIMI_ENDPOINT_PATH", DEFAULT_ENDPOINT_PATH),
            request_timeout_seconds=_to_float(
                request_timeout_seconds if request_timeout_seconds is not None else os.getenv("REQUEST_TIMEOUT_SECONDS", "30")
            ),
            max_retry=_to_int(max_retry if max_retry is not None else os.getenv("MAX_RETRY", "1")),
            temperature=_to_float(temperature if temperature is not None else os.getenv("KIMI_TEMPERATURE", "1.0")),
        )
        self._opener = opener or build_opener()

    def chat(self, messages: list[dict[str, Any]], temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self.config.temperature
        log_llm_request(messages, model=self.config.model, temperature=temp, call_label="[LLM] 大模型请求")
        payload = self._build_payload(messages, temperature=temp)
        try:
            response = self._request_json(payload)
        except Exception as exc:
            log_llm_error(exc, 1, self.config.max_retry + 1)
            raise
        content = self._extract_content(response)
        if not content or not str(content).strip():
            raise ValueError("empty response content")
        result = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        log_llm_response(result, call_label="[LLM] 大模型响应")
        return result

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        required_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        content = self.chat(messages, temperature=temperature)
        
        # Try to extract JSON from markdown code blocks
        cleaned_content = self._extract_json_from_markdown(content)
        
        try:
            data = json.loads(cleaned_content)
            log_json_parse(cleaned_content, success=True)
        except json.JSONDecodeError as exc:
            log_json_parse(content, success=False, error=str(exc))
            raise ValueError("model response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("model response JSON must be an object")
        missing = [key for key in required_keys or () if key not in data]
        if missing:
            raise ValueError(f"model response missing required keys: {missing}")
        return data
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON content from markdown code blocks."""
        import re
        
        # Try to find JSON in markdown code blocks
        patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',       # ``` ... ```
            r'`(.*?)`',                   # `...`
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Verify it's valid JSON
                try:
                    json.loads(extracted)
                    return extracted
                except json.JSONDecodeError:
                    continue
        
        # If no markdown blocks found or none contain valid JSON, return original
        return content.strip()

    def _build_payload(self, messages: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
        self._validate_messages(messages)
        return {
            REQUEST_FIELD_MAP["model"]: self.config.model,
            REQUEST_FIELD_MAP["messages"]: messages,
            REQUEST_FIELD_MAP["temperature"]: temperature,
        }

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._build_request(payload)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retry + 1):
            try:
                with self._opener(request, timeout=self.config.request_timeout_seconds) as response:
                    raw = response.read()
                if not raw:
                    raise ValueError("empty HTTP response body")
                return self._decode_json(raw)
            except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                log_llm_error(exc, attempt + 1, self.config.max_retry + 1)
                if attempt >= self.config.max_retry:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without an explicit error")

    def _build_request(self, payload: dict[str, Any]) -> urllib_request.Request:
        if not self.config.base_url:
            raise RuntimeError("KIMI_BASE_URL is not configured")
        url = self._build_url()
        headers = dict(DEFAULT_HEADERS)
        if self.config.api_key:
            headers[AUTH_HEADER_NAME] = f"{AUTH_HEADER_PREFIX}{self.config.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib_request.Request(url=url, data=body, headers=headers, method="POST")

    def _build_url(self) -> str:
        """Build the final request URL, handling duplicate path segments safely."""
        base = self.config.base_url.rstrip("/")
        path = self.config.endpoint_path.lstrip("/")
        # If base already ends with the first segment of path, avoid duplication
        base_segments = base.split("/")
        path_segments = path.split("/")
        if base_segments and path_segments and base_segments[-1] == path_segments[0]:
            path = "/".join(path_segments[1:])
        return f"{base}/{path}"

    def _decode_json(self, raw: bytes) -> dict[str, Any]:
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("HTTP response must be a JSON object")
        return decoded

    def _extract_content(self, response: dict[str, Any]) -> Any:
        for path in RESPONSE_CONTENT_PATHS:
            current: Any = response
            try:
                for part in path:
                    if isinstance(part, int):
                        current = current[part]
                    else:
                        current = current[part]
                if current is not None:
                    return current
            except (KeyError, IndexError, TypeError):
                continue
        raise ValueError("response does not contain assistant content")

    def _validate_messages(self, messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message must be a dict")
            if "role" not in message or "content" not in message:
                raise ValueError("each message must include role and content")


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid integer value: {value!r}") from exc


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid float value: {value!r}") from exc

