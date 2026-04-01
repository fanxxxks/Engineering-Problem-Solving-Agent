from eng_solver_agent.formatter import (
    format_submission_batch,
    format_submission_item,
    format_submission_output,
)


def test_formatter_preserves_question_id_and_requires_content() -> None:
    item = format_submission_item(
        question_id=7,
        reasoning_process="placeholder",
        answer="42",
    )

    assert item == {
        "question_id": 7,
        "reasoning_process": "placeholder",
        "answer": "42",
    }


def test_formatter_rejects_empty_fields() -> None:
    try:
        format_submission_item(question_id="1", reasoning_process="", answer="42")
        raise AssertionError("expected ValueError for empty reasoning_process")
    except ValueError:
        pass

    try:
        format_submission_item(question_id="1", reasoning_process="ok", answer=" ")
        raise AssertionError("expected ValueError for empty answer")
    except ValueError:
        pass


def test_formatter_supports_batch_and_single_output() -> None:
    batch = format_submission_batch(
        [
            {"question_id": "a", "reasoning_process": "r1", "answer": "x"},
            {"question_id": "b", "reasoning_process": "r2", "answer": "y"},
        ]
    )
    single = format_submission_output(
        {"question_id": "c", "reasoning_process": "r3", "answer": "z"}
    )

    assert [item["question_id"] for item in batch] == ["a", "b"]
    assert single["question_id"] == "c"
