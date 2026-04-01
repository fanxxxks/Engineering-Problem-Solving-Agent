"""Circuits solver placeholder."""

from __future__ import annotations

from eng_solver_agent.tools.circuit_tool import CircuitTool


class CircuitsSolver:
    def __init__(self) -> None:
        self.tool = CircuitTool()

    def solve(self, prompt: str) -> str:
        return self.tool.solve(prompt)
