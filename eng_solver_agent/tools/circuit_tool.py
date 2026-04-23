"""Circuit analysis tool layer for structured electrical problems."""

from __future__ import annotations

import math
from typing import Any

from eng_solver_agent.tools._math_support import ToolUnsupportedError, load_sympy, solve_linear_system


class CircuitTool:
    def __init__(self) -> None:
        self._sympy = load_sympy()

    def solve(self, expression: str) -> str:
        return f"[circuit-tool] {expression}"

    def equivalent_resistance(self, resistors: list[float], topology: str = "series") -> float:
        if not resistors:
            raise ValueError("resistors must not be empty")
        topology = topology.lower()
        if topology == "series":
            return float(sum(resistors))
        if topology == "parallel":
            if any(r == 0 for r in resistors):
                return 0.0
            return float(1.0 / sum(1.0 / float(r) for r in resistors))
        raise ValueError("topology must be 'series' or 'parallel'")

    def node_analysis(self, netlist: dict[str, Any], ground: str = "0") -> dict[str, float]:
        nodes = [str(node) for node in netlist.get("nodes", []) if str(node) != ground]
        components = netlist.get("components", [])
        if not nodes:
            return {ground: 0.0}

        index = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        rhs = [0.0 for _ in range(size)]

        for component in components:
            ctype = str(component.get("type", "")).lower()
            if ctype == "resistor":
                n1 = str(component["n1"])
                n2 = str(component["n2"])
                value = float(component["value"])
                if value == 0:
                    raise ValueError("resistor value cannot be zero")
                conductance = 1.0 / value
                self._stamp_conductance(matrix, index, n1, n2, conductance)
            elif ctype == "current_source":
                n_plus = str(component.get("n_plus", component.get("n1")))
                n_minus = str(component.get("n_minus", component.get("n2")))
                value = float(component["value"])
                self._stamp_current(rhs, index, n_plus, n_minus, value)
            else:
                raise ToolUnsupportedError(f"unsupported netlist component: {ctype}")

        voltages = solve_linear_system(matrix, rhs)
        result = {ground: 0.0}
        for node, idx in index.items():
            result[node] = voltages[idx]
        return result

    def mesh_analysis(self, resistance_matrix: list[list[Any]], source_vector: list[Any]) -> list[float]:
        if self._sympy is not None:
            matrix = self._sympy.Matrix(resistance_matrix)
            rhs = self._sympy.Matrix(source_vector)
            solution = matrix.LUsolve(rhs)
            return [float(self._sympy.N(value)) for value in solution]
        return solve_linear_system(resistance_matrix, source_vector)

    def first_order_response(
        self,
        kind: str,
        resistance: float,
        reactive: float,
        t: float,
        initial: float,
        final: float | None = None,
    ) -> dict[str, float]:
        kind_normalized = kind.strip().upper()
        if kind_normalized == "RC":
            tau = float(resistance) * float(reactive)
        elif kind_normalized == "RL":
            if resistance == 0:
                raise ValueError("resistance cannot be zero for an RL response")
            tau = float(reactive) / float(resistance)
        else:
            raise ValueError("kind must be 'RC' or 'RL'")

        final_value = float(initial if final is None else final)
        value = final_value + (float(initial) - final_value) * math.exp(-float(t) / tau)
        return {"tau": tau, "value": value}

    def nonlinear_resistor_static_dynamic_resistance(
        self, current: float, relation: str = "u=i^2+2i"
    ) -> dict[str, float]:
        relation_key = relation.replace(" ", "").lower()
        if relation_key not in {"u=i^2+2i", "u=i**2+2*i", "u=i*i+2*i"}:
            raise ToolUnsupportedError("unsupported nonlinear resistor relation")
        i_value = float(current)
        voltage = i_value * i_value + 2.0 * i_value
        if i_value == 0:
            raise ValueError("static resistance is undefined when current is zero")
        static_resistance = voltage / i_value
        dynamic_resistance = 2.0 * i_value + 2.0
        return {
            "static_resistance": static_resistance,
            "dynamic_resistance": dynamic_resistance,
        }

    def rlc_series_underdamped_resistance_range(self, inductance: float, capacitance: float) -> dict[str, float]:
        l_value = float(inductance)
        c_value = float(capacitance)
        if l_value <= 0 or c_value <= 0:
            raise ValueError("inductance and capacitance must be positive")
        upper = 2.0 * math.sqrt(l_value / c_value)
        return {"lower_exclusive": 0.0, "upper_exclusive": upper}

    def _stamp_conductance(
        self,
        matrix: list[list[float]],
        index: dict[str, int],
        n1: str,
        n2: str,
        conductance: float,
    ) -> None:
        i = index.get(n1)
        j = index.get(n2)
        if i is not None:
            matrix[i][i] += conductance
        if j is not None:
            matrix[j][j] += conductance
        if i is not None and j is not None:
            matrix[i][j] -= conductance
            matrix[j][i] -= conductance

    def _stamp_current(
        self,
        rhs: list[float],
        index: dict[str, int],
        n_plus: str,
        n_minus: str,
        value: float,
    ) -> None:
        plus = index.get(n_plus)
        minus = index.get(n_minus)
        if plus is not None:
            rhs[plus] -= value
        if minus is not None:
            rhs[minus] += value
