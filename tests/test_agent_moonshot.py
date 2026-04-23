import os
from pathlib import Path

from eng_solver_agent.agent import EngineeringSolverAgent


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("no fake responses left")
        return _FakeResponse(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeOpenAI:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _FakeChat(responses)


def test_solve_uses_moonshot_when_key_present() -> None:
    old_key = os.environ.get("MOONSHOT_API_KEY")
    try:
        os.environ["MOONSHOT_API_KEY"] = "test-key"
        client = _FakeOpenAI(['{"reasoning_process":"步骤","answer":"42"}'])
        agent = EngineeringSolverAgent(openai_client=client)
        result = agent.solve({"question_id": "m-1", "question": "2+40=?"})
        assert result["question_id"] == "m-1"
        assert result["answer"] == "42"
    finally:
        if old_key is None:
            os.environ.pop("MOONSHOT_API_KEY", None)
        else:
            os.environ["MOONSHOT_API_KEY"] = old_key


def test_run_reasoning_loop_executes_python_tool() -> None:
    old_key = os.environ.get("MOONSHOT_API_KEY")
    try:
        os.environ["MOONSHOT_API_KEY"] = "test-key"
        client = _FakeOpenAI(
            [
                "先计算。\n```python\nprint(1+2)\n```",
                '{"reasoning_process":"根据执行结果得到答案","answer":"3"}',
            ]
        )
        agent = EngineeringSolverAgent(openai_client=client)
        result = agent.solve({"question_id": "m-2", "question": "1+2=?"})
        assert result["answer"] == "3"
        assert len(client.chat.completions.calls) == 2
    finally:
        if old_key is None:
            os.environ.pop("MOONSHOT_API_KEY", None)
        else:
            os.environ["MOONSHOT_API_KEY"] = old_key


def test_parse_multimodal_input_adds_image_url(tmp_path: Path) -> None:
    agent = EngineeringSolverAgent()
    image_path = tmp_path / "q.jpg"
    image_path.write_bytes(b"fake-image")
    content = agent.parse_multimodal_input({"question": "看图求解", "image_path": str(image_path)})
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
