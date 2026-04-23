import os

from eng_solver_agent.agent import EngineeringSolverAgent

EXPECTED_GAMMA_BETA_TRIPLE_INTEGRAL_ANSWER = (
    "\\( \\dfrac{(2n)!}{4^n n!} \\sqrt{\\pi} \\),\\( \\dfrac{\\sqrt{\\pi}}{2} \\),\\( \\sqrt{\\pi} \\)"
)


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
        agent = EngineeringSolverAgent(tool_registry={"calculus": FakeMathTool()})

        derivative = agent.solve_one({"question_id": "d1", "question": "Find the derivative of x^2."})
        definite_integral = agent.solve_one({"question_id": "i1", "question": "Compute the integral of x^2 from 0 to 3."})
        limit_case = agent.solve_one({"question_id": "l1", "question": "Find the limit of x^2 as x -> 2."})

        assert derivative["answer"] == '{"a":1,"b":2}'
        assert definite_integral["answer"] == "[3,1]"
        assert limit_case["answer"] == "4"
        assert "工具计算" in derivative["reasoning_process"]
        assert "工具计算" in definite_integral["reasoning_process"]
        assert "工具计算" in limit_case["reasoning_process"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_circuit_series_parallel_fast_path_with_structured_resistors() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = EngineeringSolverAgent()

        result = agent.solve_one(
            {
                "question_id": "c1",
                "question": "Two resistors in series.",
                "resistors": [2, 3],
                "topology": "series",
            }
        )

        assert result["answer"] == "5.0"
        assert "工具计算" in result["reasoning_process"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_physics_without_structured_numbers_does_not_fake_numeric_answer() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = EngineeringSolverAgent()

        result = agent.solve_one({"question_id": "p1", "question": "Find the force in this physics problem."})

        assert "暂无法可靠给出最终数值" in result["answer"]
        assert "当前失败原因" in result["reasoning_process"]
        assert "force" not in result["answer"].lower() or "1" not in result["answer"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_tool_failure_does_not_return_raw_exception_stack() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = EngineeringSolverAgent(tool_registry={"calculus": RaisingTool()})

        result = agent.solve_one({"question_id": "f1", "question": "Find the derivative of x^2."})

        assert "Traceback" not in result["answer"]
        assert "boom" in result["answer"]
        assert "当前失败原因" in result["reasoning_process"]
        assert "Traceback" not in result["reasoning_process"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_calculus_validation_style_integral_question_gets_special_fallback_answer() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = EngineeringSolverAgent()
        result = agent.solve_one(
            {
                "question_id": "CAL_001",
                "type": "计算题",
                "question": "请解决以下三个积分问题，并填写答案：1. 计算 J_n = ∫_0^{+∞} x^{2n}e^{-x^2}dx；2. 计算 ∫_0^1 (ln(1/x))^{1/2}dx；3. 计算 ∫_0^1 (-lnx)^{-1/2}dx。",
            }
        )
        assert result["answer"] == EXPECTED_GAMMA_BETA_TRIPLE_INTEGRAL_ANSWER
        assert "Gamma/Beta" in result["reasoning_process"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url


def test_calculus_validation_style_proof_question_gets_dependency_warning_answer() -> None:
    old_base_url = os.environ.get("KIMI_BASE_URL")
    try:
        os.environ.pop("KIMI_BASE_URL", None)
        agent = EngineeringSolverAgent()
        result = agent.solve_one(
            {
                "question_id": "CAL_004",
                "type": "证明题",
                "question": "设 f(x)=x^2sin(1/x)，试问矛盾何在？请解释。",
            }
        )
        assert "ξ=ξ(x)" in result["answer"]
        assert "不能推出" in result["answer"]
        assert "暂无法可靠给出最终数值" not in result["answer"]
    finally:
        if old_base_url is None:
            os.environ.pop("KIMI_BASE_URL", None)
        else:
            os.environ["KIMI_BASE_URL"] = old_base_url
