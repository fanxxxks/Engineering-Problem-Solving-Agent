from eng_solver_agent.tools._math_support import ToolUnsupportedError, load_sympy
from eng_solver_agent.tools.calculus_tool import CalculusTool
from tests._helpers import approx_equal, assert_raises


def test_calculus_tool_diff_integrate_limit() -> None:
    tool = CalculusTool()

    assert tool.diff("x**3 + 2*x") == "3*x**2 + 2"
    assert approx_equal(tool.diff("x**2", at=2), 4.0)
    assert tool.integrate("x**2") == "1/3*x**3"
    assert approx_equal(tool.integrate("x**2", lower=0, upper=3), 9.0)
    assert approx_equal(tool.limit("x**2", point=2), 4.0)


def test_calculus_tool_critical_points_and_taylor_series() -> None:
    tool = CalculusTool()

    critical_points = tool.critical_points("x**2 + 2*x + 1")
    assert len(critical_points) == 1
    assert approx_equal(critical_points[0], -1.0)

    series = tool.taylor_series("x**2", center=0, order=2)
    assert series["coefficients"] == [0.0, 0.0, 1.0]
    assert "x**2" in series["series"]


def test_calculus_tool_reports_unsupported_symbolic_input_without_sympy() -> None:
    if load_sympy() is not None:
        return
    tool = CalculusTool()

    assert_raises(ToolUnsupportedError, tool.diff, "sin(x)")
