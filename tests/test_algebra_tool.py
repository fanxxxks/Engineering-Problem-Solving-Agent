from eng_solver_agent.tools._math_support import ToolUnsupportedError, load_sympy
from eng_solver_agent.tools.algebra_tool import AlgebraTool
from tests._helpers import approx_equal, assert_raises


def test_algebra_tool_solves_linear_system() -> None:
    tool = AlgebraTool()

    solution = tool.solve_linear_system([[2, 1], [1, -1]], [5, 1])

    assert approx_equal(solution[0], 2.0)
    assert approx_equal(solution[1], 1.0)


def test_algebra_tool_matrix_operations() -> None:
    tool = AlgebraTool()

    assert approx_equal(tool.determinant([[4, 7], [2, 6]]), 10.0)
    inverse = tool.matrix_inverse([[4, 7], [2, 6]])
    assert approx_equal(inverse[0][0], 0.6)
    assert approx_equal(inverse[0][1], -0.7)
    assert approx_equal(inverse[1][0], -0.2)
    assert approx_equal(inverse[1][1], 0.4)
    assert tool.rank([[1, 2], [2, 4]]) == 1


def test_algebra_tool_eigenvalues_and_simplify() -> None:
    tool = AlgebraTool()

    eigenvalues = sorted(float(value) for value in tool.eigenvalues([[2, 0], [0, 3]]))
    assert approx_equal(eigenvalues[0], 2.0)
    assert approx_equal(eigenvalues[1], 3.0)
    assert tool.simplify("2*x + 3*x - x") == "4*x"


def test_algebra_tool_reports_unsupported_symbolic_input_without_sympy() -> None:
    if load_sympy() is not None:
        return
    tool = AlgebraTool()

    assert_raises(ToolUnsupportedError, tool.simplify, "sin(x)")
