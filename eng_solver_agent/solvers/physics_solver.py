"""Physics solver placeholder."""

from __future__ import annotations

from eng_solver_agent.tools.physics_tool import PhysicsTool


class PhysicsSolver:
    def __init__(self) -> None:
        self.tool = PhysicsTool()

    def solve(self, prompt: str) -> str:
        return self.tool.solve(prompt)
