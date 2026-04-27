import os

from eng_solver_agent.unified_agent import UnifiedAgent


class FakeMathTool:
    def diff(self, expression: str, var: str = "x", order: int = 1, at=None):
        return {"b": 2, "a": 1}

    def integrate(self, expression: str, var: str = "x", lower=None, upper=None):
        return [3, 1]

    def limit(self, expression: str, var: str = "x", point=0, direction: str = "both"):
        return 4

    def critical_points(self, expression: str, var: str = "x"):
        return []

    def taylor_series(self, expression: str, var: str = "x", center=0, order=5):
        return {"series": "x"}


class RaisingTool:
    def diff(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_calculus_natural_language_fast_paths_work_without_kimi() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = UnifiedAgent(kimi_client=None, tool_registry={"calculus": FakeMathTool()})

        derivative = agent.solve_one({"question_id": "d1", "question": "Find the derivative of x^2.", "subject": "calculus"}, mode="tool_only")
        definite_integral = agent.solve_one({"question_id": "i1", "question": "Compute the integral of x^2 from 0 to 3.", "subject": "calculus"}, mode="tool_only")
        limit_case = agent.solve_one({"question_id": "l1", "question": "Find the limit of x^2 as x -> 2.", "subject": "calculus"}, mode="tool_only")

        assert derivative["answer"] == '{"a":1,"b":2}'
        assert definite_integral["answer"] == "[3,1]"
        assert limit_case["answer"] == "4"
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_circuit_series_parallel_fast_path_with_structured_resistors() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = UnifiedAgent(kimi_client=None)

        result = agent.solve_one(
            {
                "question_id": "c1",
                "question": "Two resistors in series.",
                "resistors": [2, 3],
                "topology": "series",
            },
            mode="tool_only",
        )

        assert result["answer"] == "5.0"
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_physics_without_structured_numbers_does_not_fake_numeric_answer() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = UnifiedAgent(kimi_client=None)

        result = agent.solve_one({"question_id": "p1", "question": "Find the force in this physics problem."}, mode="tool_only")

        assert "暂无法计算" in result["answer"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_tool_failure_does_not_return_raw_exception_stack() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = UnifiedAgent(kimi_client=None, tool_registry={"calculus": RaisingTool()})

        result = agent.solve_one({"question_id": "f1", "question": "Find the derivative of x^2.", "subject": "calculus"}, mode="tool_only")

        assert "Traceback" not in result["answer"]
        assert "boom" in result["answer"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url
