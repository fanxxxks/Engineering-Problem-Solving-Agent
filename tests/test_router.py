from eng_solver_agent.router import QuestionRouter


def test_router_detects_calculus_prompt() -> None:
    router = QuestionRouter()

    assert router.route({"question": "Find the derivative of sin(x)."}) == "calculus"


def test_router_prefers_higher_priority_on_conflict() -> None:
    router = QuestionRouter()

    decision = router.route_with_confidence(
        {"question": "Analyze a circuit using matrix methods and voltage nodes."}
    )

    assert decision.subject == "circuits"
    assert decision.confidence >= 0.72
    assert "circuit" in decision.matched_rules


def test_router_falls_back_to_physics() -> None:
    router = QuestionRouter()

    assert router.route({"question": "Explain this engineering problem."}) == "physics"


def test_router_detects_chinese_calculus_prompt() -> None:
    router = QuestionRouter()

    assert router.route({"question_id": "CAL_001", "question": "计算定积分并求极限。"}) == "calculus"
