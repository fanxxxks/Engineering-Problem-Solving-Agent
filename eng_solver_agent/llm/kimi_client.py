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


DEFAULT_ENDPOINT_PATH = os.getenv("KIMI_ENDPOINT_PATH", "/v1/chat/completions")
REQUEST_FIELD_MAP = {
    "model": "model",
    "messages": "messages",
    "temperature": "temperature",
}
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
AUTH_HEADER_NAME = "Authorization"
AUTH_HEADER_PREFIX = "Bearer "
RESPONSE_CONTENT_PATHS = (
    ("choices", 0, "message", "content"),
    ("choices", 0, "text"),
    ("output_text",),
    ("content",),
)


@dataclass(frozen=True)
class KimiConfig:
    api_key: str
    base_url: str
    model: str
    endpoint_path: str
    request_timeout_seconds: float
    max_retry: int


class KimiClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        endpoint_path: str | None = None,
        request_timeout_seconds: float | None = None,
        max_retry: int | None = None,
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
        )
        self._opener = opener or urllib_request.urlopen

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> str:
        payload = self._build_payload(messages, temperature=temperature)
        response = self._request_json(payload)
        content = self._extract_content(response)
        if not content or not str(content).strip():
            raise ValueError("empty response content")
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        required_keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        content = self.chat(messages, temperature=temperature)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("model response JSON must be an object")
        missing = [key for key in required_keys or () if key not in data]
        if missing:
            raise ValueError(f"model response missing required keys: {missing}")
        return data

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
                if attempt >= self.config.max_retry:
                    raise
                time.sleep(0)
        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without an explicit error")

    def _build_request(self, payload: dict[str, Any]) -> urllib_request.Request:
        if not self.config.base_url:
            raise RuntimeError("KIMI_BASE_URL is not configured")
        url = urljoin(self.config.base_url.rstrip("/") + "/", self.config.endpoint_path.lstrip("/"))
        headers = dict(DEFAULT_HEADERS)
        if self.config.api_key:
            headers[AUTH_HEADER_NAME] = f"{AUTH_HEADER_PREFIX}{self.config.api_key}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return urllib_request.Request(url=url, data=body, headers=headers, method="POST")

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

