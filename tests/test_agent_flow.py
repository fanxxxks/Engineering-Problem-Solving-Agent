import os

from eng_solver_agent.agent import EngineeringSolverAgent
from eng_solver_agent.retrieval.kb_loader import KnowledgeBaseLoader
from eng_solver_agent.retrieval.retriever import Retriever


class RecordingFakeKimiClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat_json(self, messages, temperature: float = 0.0, required_keys=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "required_keys": tuple(required_keys or ()),
            }
        )
        if not self.responses:
            raise AssertionError("no more fake responses available")
        return self.responses.pop(0)


def test_agent_two_stage_flow_with_fake_kimi_client() -> None:
    client = RecordingFakeKimiClient(
        [
            {
                "subject": "calculus",
                "topic": "differentiation",
                "knowns": ["x**2"],
                "unknowns": ["derivative"],
                "equations_or_theorems": ["d/dx x**n = n*x**(n-1)"],
                "should_use_tool": True,
                "target_form": "derivative expression",
                "possible_traps": ["power rule"],
            },
            {
                "reasoning_process": "Knowns -> tool result -> final result.",
                "answer": "2*x",
            },
        ]
    )
    agent = EngineeringSolverAgent(
        kimi_client=client,
    )

    result = agent.solve_one(
        {
            "question_id": "q-1",
            "question": "Differentiate x**2.",
            "expression": "x**2",
        }
    )

    assert result["question_id"] == "q-1"
    assert result["reasoning_process"] == "Knowns -> tool result -> final result."
    assert result["answer"] == "2*x"
    assert len(client.calls) == 2
    assert "subject" in client.calls[0]["required_keys"]
    assert "reasoning_process" in client.calls[1]["required_keys"]


def test_agent_fallback_flow_without_kimi_configuration() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = EngineeringSolverAgent()

        cases = [
            (
                {"question_id": 101, "question": "Differentiate x**2.", "expression": "x**2"},
                "question_id",
            ),
            (
                {
                    "question_id": 102,
                    "question": "Find determinant.",
                    "matrix": [[1, 2], [3, 4]],
                },
                "question_id",
            ),
            (
                {
                    "question_id": 103,
                    "question": "Two resistors in series.",
                    "resistors": [2, 3],
                    "topology": "series",
                },
                "question_id",
            ),
            (
                {
                    "question_id": 104,
                    "question": "Find force.",
                    "relation": "newton_second_law",
                    "knowns": {"m": 2, "a": 5},
                    "target": "F",
                },
                "question_id",
            ),
        ]

        for question, key_name in cases:
            result = agent.solve_one(question)
            assert result[key_name] == question[key_name]
            assert set(result) == {"question_id", "reasoning_process", "answer"}
            assert str(result["reasoning_process"]).strip()
            assert str(result["answer"]).strip()
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_agent_with_retriever_still_solves_one_question() -> None:
    loader = KnowledgeBaseLoader()
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eng_solver_agent", "retrieval")
    retriever = Retriever(
        formula_cards=loader.load(os.path.join(root, "formula_cards.json")),
        solved_examples=loader.load(os.path.join(root, "solved_examples.jsonl")),
    )

    agent = EngineeringSolverAgent(retriever=retriever)
    result = agent.solve_one({"question_id": "r1", "question": "Find the derivative of x^2."})

    assert result["question_id"] == "r1"
    assert str(result["reasoning_process"]).strip()
    assert str(result["answer"]).strip()


def test_agent_empty_retriever_does_not_crash_on_unknown_query() -> None:
    agent = EngineeringSolverAgent(retriever=Retriever(formula_cards=[], solved_examples=[]))

    result = agent.solve_one({"question_id": "r2", "question": "Some unknown engineering question."})

    assert result["question_id"] == "r2"
    assert set(result) == {"question_id", "reasoning_process", "answer"}
