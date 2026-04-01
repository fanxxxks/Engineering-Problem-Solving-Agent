from eng_solver_agent.tools.unit_tool import UnitTool


def test_unit_tool_dimensions_and_compatibility() -> None:
    tool = UnitTool()

    assert tool.dimension_of("V") == {"M": 1, "L": 2, "T": -3, "I": -1}
    assert tool.dimension_of("Ω") == {"M": 1, "L": 2, "T": -3, "I": -2}
    assert tool.compatible("m", "m") is True
    assert tool.compatible("m", "s") is False


def test_unit_tool_reasoning_process_check() -> None:
    tool = UnitTool()

    report = tool.check_reasoning_process("The voltage is 5 V and current is 2 A.")

    assert report["has_known_units"] is True
    assert "V" in report["recognized_units"]
    assert "A" in report["recognized_units"]
