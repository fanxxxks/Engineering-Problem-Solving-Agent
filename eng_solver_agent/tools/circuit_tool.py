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

        voltage_sources: list[dict[str, Any]] = []
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
            elif ctype == "voltage_source":
                voltage_sources.append(component)
            else:
                raise ToolUnsupportedError(f"unsupported netlist component: {ctype}")

        self._apply_voltage_sources(voltage_sources, matrix, rhs, index, ground)

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

    def _apply_voltage_sources(
        self,
        sources: list[dict[str, Any]],
        matrix: list[list[float]],
        rhs: list[float],
        index: dict[str, int],
        ground: str,
    ) -> None:
        for source in sources:
            n_plus = str(source.get("n_plus", source.get("n1")))
            n_minus = str(source.get("n_minus", source.get("n2")))
            value = float(source["value"])
            plus_idx = index.get(n_plus)
            minus_idx = index.get(n_minus)

            if n_minus == ground and plus_idx is not None:
                matrix[plus_idx] = [0.0] * len(matrix)
                matrix[plus_idx][plus_idx] = 1.0
                rhs[plus_idx] = value
            elif n_plus == ground and minus_idx is not None:
                matrix[minus_idx] = [0.0] * len(matrix)
                matrix[minus_idx][minus_idx] = 1.0
                rhs[minus_idx] = -value
            elif plus_idx is not None and minus_idx is not None:
                approx_resistance = 1e-9
                conductance = 1.0 / approx_resistance
                self._stamp_conductance(matrix, index, n_plus, n_minus, conductance)
                self._stamp_current(rhs, index, n_plus, n_minus, value / approx_resistance)
            else:
                raise ToolUnsupportedError("voltage source with floating node is not supported")

