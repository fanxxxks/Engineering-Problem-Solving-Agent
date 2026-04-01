import json
import os

from eng_solver_agent.llm.kimi_client import KimiClient
from tests._helpers import assert_raises


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def make_opener(body: bytes):
    requests = []

    def opener(request, timeout=None):
        requests.append((request, timeout))
        return FakeResponse(body)

    opener.requests = requests  # type: ignore[attr-defined]
    return opener


def test_kimi_client_reads_env_and_parses_json_content() -> None:
    old_env = {key: os.environ.get(key) for key in ["KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL", "REQUEST_TIMEOUT_SECONDS", "MAX_RETRY"]}
    try:
        os.environ["KIMI_API_KEY"] = "env-key"
        os.environ["KIMI_BASE_URL"] = "https://example.test"
        os.environ["KIMI_MODEL"] = "kimi-test"
        os.environ["REQUEST_TIMEOUT_SECONDS"] = "12"
        os.environ["MAX_RETRY"] = "2"
        opener = make_opener(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"reasoning_process": "step", "answer": "42"})
                            }
                        }
                    ]
                }
            ).encode("utf-8")
        )
        client = KimiClient(opener=opener)
        data = client.chat_json([{"role": "user", "content": "solve"}])

        assert client.config.api_key == "env-key"
        assert client.config.base_url == "https://example.test"
        assert client.config.model == "kimi-test"
        assert client.config.request_timeout_seconds == 12.0
        assert client.config.max_retry == 2
        assert data == {"reasoning_process": "step", "answer": "42"}
        assert opener.requests[0][1] == 12.0
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_kimi_client_rejects_empty_response_and_invalid_json() -> None:
    empty_opener = make_opener(
        json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")
    )
    invalid_json_opener = make_opener(
        json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode("utf-8")
    )
    client = KimiClient(base_url="https://example.test", opener=empty_opener)
    assert_raises(ValueError, client.chat, [{"role": "user", "content": "solve"}])

    client = KimiClient(base_url="https://example.test", opener=invalid_json_opener)
    assert_raises(ValueError, client.chat_json, [{"role": "user", "content": "solve"}])


def test_kimi_client_rejects_missing_fields_when_required() -> None:
    missing_fields_opener = make_opener(
        json.dumps({"choices": [{"message": {"content": json.dumps({"reasoning_process": "step"})}}]}).encode("utf-8")
    )
    client = KimiClient(base_url="https://example.test", opener=missing_fields_opener)
    assert_raises(
        ValueError,
        client.chat_json,
        [{"role": "user", "content": "solve"}],
        0.0,
        ["reasoning_process", "answer"],
    )
