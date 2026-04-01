from eng_solver_agent.agent import EngineeringSolverAgent


def test_agent_solve_one_returns_competition_shape() -> None:
    agent = EngineeringSolverAgent()

    result = agent.solve_one({"question_id": "q-1", "prompt": "Find voltage."})

    assert result["question_id"] == "q-1"
    assert set(result) == {"question_id", "reasoning_process", "answer"}


def test_agent_solve_handles_multiple_questions() -> None:
    agent = EngineeringSolverAgent()

    results = agent.solve(
        [
            {"question_id": "q-1", "prompt": "Find voltage."},
            {"question_id": "q-2", "prompt": "Find the derivative of x^2."},
        ]
    )

    assert len(results) == 2
    assert [item["question_id"] for item in results] == ["q-1", "q-2"]
