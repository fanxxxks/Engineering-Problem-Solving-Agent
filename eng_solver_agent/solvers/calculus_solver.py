"""Calculus solver placeholder."""

from __future__ import annotations

from eng_solver_agent.tools.calculus_tool import CalculusTool


class CalculusSolver:
    def __init__(self) -> None:
        self.tool = CalculusTool()

    def solve(self, prompt: str) -> str:
        return self.tool.solve(prompt)
