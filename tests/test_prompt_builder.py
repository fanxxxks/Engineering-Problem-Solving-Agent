from eng_solver_agent.llm.prompt_builder import (
    ANALYZE_REQUIRED_FIELDS,
    DRAFT_REQUIRED_FIELDS,
    build_analyze_messages,
    build_analyze_prompt,
    build_draft_messages,
    build_draft_prompt,
)


def test_analyze_prompt_contains_required_fields_for_each_subject() -> None:
    cases = {
        "physics": "kinematics",
        "circuits": "node analysis",
        "linalg": "eigenvalues",
        "calculus": "limits",
    }

    for subject, focus_word in cases.items():
        prompt = build_analyze_prompt(
            {"question_id": "q1", "question": "demo question", "subject": subject},
            subject=subject,
        )
        for field in ANALYZE_REQUIRED_FIELDS:
            assert field in prompt
        assert "strict JSON only" in prompt
        assert focus_word in prompt


def test_draft_prompt_contains_required_fields_and_style_rules() -> None:
    prompt = build_draft_prompt(
        {"question_id": "q1", "question": "demo question", "subject": "physics"},
        {"subject": "physics", "topic": "kinematics"},
        tool_results=[{"tool_name": "physics_tool", "output": "v = 10"}],
        subject="physics",
    )

    for field in DRAFT_REQUIRED_FIELDS:
        assert field in prompt
    assert "knowns and unknowns" in prompt
    assert "formula, theorem" in prompt
    assert "substitution or derivation" in prompt
    assert "result and any necessary units" in prompt


def test_message_builders_wrap_prompt_text() -> None:
    analyze_messages = build_analyze_messages({"question": "Find x"}, subject="calculus")
    draft_messages = build_draft_messages(
        {"question": "Find x"},
        {"subject": "calculus", "topic": "derivative"},
        tool_results=[],
        subject="calculus",
    )

    assert analyze_messages[0]["role"] == "system"
    assert analyze_messages[1]["role"] == "user"
    assert "strict JSON only" in analyze_messages[1]["content"]
    assert draft_messages[1]["role"] == "user"
    assert "reasoning_process" in draft_messages[1]["content"]
