from eng_solver_agent.tools._math_support import ToolUnsupportedError
from eng_solver_agent.tools.physics_tool import PhysicsTool
from tests._helpers import approx_equal, assert_raises


def test_physics_tool_uniform_acceleration() -> None:
    tool = PhysicsTool()

    assert approx_equal(tool.solve_relation("uniform_acceleration", {"v0": 2, "a": 3, "t": 4}, "v"), 14.0)


def test_physics_tool_newton_work_momentum() -> None:
    tool = PhysicsTool()

    assert approx_equal(tool.solve_relation("newton_second_law", {"m": 2, "a": 5}, "F"), 10.0)
    assert approx_equal(tool.solve_relation("work_energy", {"F": 10, "s": 2, "theta": 0}, "W"), 20.0)
    assert approx_equal(tool.solve_relation("momentum", {"m": 3, "v": 4}, "p"), 12.0)


def test_physics_tool_reports_unsupported_relation() -> None:
    tool = PhysicsTool()

    assert_raises(ToolUnsupportedError, tool.solve_relation, "drag", {}, "v")
