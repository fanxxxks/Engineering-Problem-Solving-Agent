"""Minimal script to exercise the agent locally."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eng_solver_agent.agent import EngineeringSolverAgent


def main() -> None:
    agent = EngineeringSolverAgent()
    single = agent.solve_one(
        {
            "question_id": "demo-1",
            "prompt": "Find the derivative of x^2.",
        }
    )
    batch = agent.solve(
        [
            {
                "question_id": "demo-2",
                "prompt": "Two resistors of 2 ohm and 3 ohm are in series.",
                "resistors": [2, 3],
                "topology": "series",
            },
            {
                "question_id": "demo-3",
                "question": "Find the limit of x^2 as x -> 2.",
                "expression": "x^2",
                "point": 2,
                "subject": "calculus",
            },
        ]
    )
    print(single)
    for item in batch:
        print(item)


if __name__ == "__main__":
    main()
