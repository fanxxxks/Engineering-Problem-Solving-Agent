from eng_solver_agent.tools.circuit_tool import CircuitTool
from tests._helpers import approx_equal


def test_circuit_tool_equivalent_resistance() -> None:
    tool = CircuitTool()

    assert approx_equal(tool.equivalent_resistance([2, 3, 5], topology="series"), 10.0)
    assert approx_equal(tool.equivalent_resistance([2, 2], topology="parallel"), 1.0)


def test_circuit_tool_node_analysis() -> None:
    tool = CircuitTool()

    result = tool.node_analysis(
        {
            "nodes": ["1"],
            "components": [
                {"type": "resistor", "n1": "1", "n2": "0", "value": 10},
                {"type": "current_source", "n_plus": "0", "n_minus": "1", "value": 2},
            ],
        }
    )

    assert approx_equal(result["1"], 20.0)
    assert approx_equal(result["0"], 0.0)


def test_circuit_tool_mesh_and_first_order_response() -> None:
    tool = CircuitTool()

    mesh = tool.mesh_analysis([[3, 1], [1, 2]], [9, 8])
    assert approx_equal(mesh[0], 2.0)
    assert approx_equal(mesh[1], 3.0)

    response = tool.first_order_response("RC", resistance=2, reactive=3, t=6, initial=5, final=1)
    assert approx_equal(response["tau"], 6.0)
    assert response["value"] < 5.0
    assert response["value"] > 1.0


def test_circuit_tool_nonlinear_resistor_and_rlc_underdamped_range() -> None:
    tool = CircuitTool()

    nonlinear = tool.nonlinear_resistor_static_dynamic_resistance(current=1.0)
    assert approx_equal(nonlinear["static_resistance"], 3.0)
    assert approx_equal(nonlinear["dynamic_resistance"], 4.0)

    r_range = tool.rlc_series_underdamped_resistance_range(inductance=10e-3, capacitance=1e-6)
    assert approx_equal(r_range["lower_exclusive"], 0.0)
    assert approx_equal(r_range["upper_exclusive"], 200.0)
