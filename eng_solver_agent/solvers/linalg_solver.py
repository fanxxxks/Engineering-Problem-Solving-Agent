"""Linear algebra solver placeholder."""

from __future__ import annotations

from eng_solver_agent.tools.algebra_tool import AlgebraTool


class LinearAlgebraSolver:
    def __init__(self) -> None:
        self.tool = AlgebraTool()

    def solve(self, prompt: str) -> str:
        return self.tool.solve(prompt)
