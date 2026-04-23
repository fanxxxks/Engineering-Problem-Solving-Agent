import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eng_solver_agent.eval.local_eval import compare_answers, load_dev_set, run_local_eval


def test_compare_answers_supports_exact_and_numeric_tolerance() -> None:
    exact, numeric = compare_answers("2*x", "2*x")
    assert exact is True
    assert numeric is False

    exact, numeric = compare_answers("10.0000001", "10")
    assert exact is True
    assert numeric is True

    exact, numeric = compare_answers("10", "11")
    assert exact is False
    assert numeric is False


def test_load_dev_set_reads_real_dev_file() -> None:
    dev_path = Path(__file__).resolve().parents[1] / "data" / "dev" / "dev.json"
    questions = load_dev_set(dev_path)

    assert len(questions) == 2
    assert [item["question_id"] for item in questions] == ["PHY_001", "CIR_001"]


def test_run_local_eval_writes_predictions_and_report() -> None:
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        dev_path = base / "dev.json"
        predictions_path = base / "predictions.json"
        report_path = base / "report.json"
        dev_path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "question_id": "physics-1",
                            "question": "A 2 kg object has a net force of 10 N. Find the acceleration.",
                            "subject": "physics",
                            "relation": "newton_second_law",
                            "knowns": {"m": 2, "a": 5},
                            "target": "F",
                            "gold_answer": "10",
                        },
                        {
                            "question_id": "circuits-1",
                            "question": "Two resistors of 2 ohm and 3 ohm are in series. Find the equivalent resistance.",
                            "subject": "circuits",
                            "resistors": [2, 3],
                            "topology": "series",
                            "gold_answer": "5",
                        },
                        {
                            "question_id": "linalg-1",
                            "question": "Find the determinant of [[4,7],[2,6]].",
                            "subject": "linalg",
                            "matrix": [[4, 7], [2, 6]],
                            "gold_answer": "10",
                        },
                        {
                            "question_id": "calculus-1",
                            "question": "Find the derivative of x^2.",
                            "subject": "calculus",
                            "expression": "x^2",
                            "gold_answer": "2*x",
                        },
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        predictions, report = run_local_eval(dev_path, predictions_path, report_path)

        assert predictions_path.exists()
        assert report_path.exists()
        assert len(predictions) == 4
        assert report["total_count"] == 4
        assert report["exact_match_count"] == 4
        assert report["exact_match_accuracy"] == 1.0
        assert set(report["per_subject"]) == {"physics", "circuits", "linalg", "calculus"}
        assert all(item["question_id"] for item in predictions)
        assert all("prediction" in item for item in predictions)
        assert all("reasoning_process" in item for item in predictions)


def test_run_local_eval_handles_fallback_like_entries() -> None:
    with TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        dev_path = base / "dev.json"
        predictions_path = base / "predictions.json"
        report_path = base / "report.json"
        dev_path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "question_id": "physics-fallback",
                            "question": "Find the force in this physics problem.",
                            "subject": "physics",
                            "gold_answer": "暂无法可靠给出最终数值: Physics fast path needs structured knowns, relation, and target to avoid guessing.",
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        _, report = run_local_eval(dev_path, predictions_path, report_path)

        assert report["fallback_like_count"] >= 1
        assert report["failure_count"] == 0
