"""Local evaluation helpers for the development set."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FALLBACK_HINTS = (
    "暂无法可靠给出最终数值",
    "当前 fallback 未能完成精确求解",
)


@dataclass(frozen=True)
class LocalEvalPaths:
    dev_path: Path
    predictions_path: Path
    report_path: Path


def load_dev_set(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"dev set not found: {candidate}")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return _ensure_questions(data, candidate)
    if isinstance(data, dict):
        if "questions" in data and isinstance(data["questions"], list):
            return _ensure_questions(data["questions"], candidate)
        if "items" in data and isinstance(data["items"], list):
            return _ensure_questions(data["items"], candidate)
    raise ValueError(f"dev set must be a list or contain a questions/items array: {candidate}")


def run_local_eval(
    dev_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
    agent: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions = load_dev_set(dev_path)
    if agent is None:
        from eng_solver_agent.unified_agent import UnifiedAgent
        agent = UnifiedAgent()
    solver = agent
    predictions: list[dict[str, Any]] = []
    for question in questions:
        prediction = solver.solve_one(question)
        exact_match, numeric_match = compare_answers(prediction.get("answer", ""), question.get("gold_answer", ""))
        prediction_record = {
            "question_id": question.get("question_id"),
            "subject": question.get("subject"),
            "question": question.get("question"),
            "gold_answer": question.get("gold_answer"),
            "prediction": prediction.get("answer", ""),
            "reasoning_process": prediction.get("reasoning_process", ""),
            "exact_match": exact_match,
            "numeric_match": numeric_match,
            "answered": bool(str(prediction.get("answer", "")).strip()),
            "fallback_like": _is_fallback_like(prediction),
        }
        predictions.append(prediction_record)

    report = evaluate_dev_set(predictions)
    _write_json(predictions_path, predictions)
    _write_json(report_path, report)
    return predictions, report


def evaluate_dev_set(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = len(predictions)
    answered_count = sum(1 for item in predictions if item.get("answered"))
    exact_match_count = sum(1 for item in predictions if item.get("exact_match"))
    fallback_like_count = sum(1 for item in predictions if item.get("fallback_like"))
    failure_count = total_count - answered_count
    average_answer_length = _average(len(str(item.get("prediction", ""))) for item in predictions)
    average_reasoning_length = _average(len(str(item.get("reasoning_process", ""))) for item in predictions)

    per_subject_totals: dict[str, int] = defaultdict(int)
    per_subject_correct: dict[str, int] = defaultdict(int)
    for item in predictions:
        subject = str(item.get("subject") or "unknown")
        per_subject_totals[subject] += 1
        if item.get("exact_match"):
            per_subject_correct[subject] += 1

    per_subject = {}
    for subject, count in per_subject_totals.items():
        correct = per_subject_correct[subject]
        per_subject[subject] = {
            "count": count,
            "exact_match_count": correct,
            "accuracy": correct / count if count else 0.0,
        }

    return {
        "total_count": total_count,
        "answered_count": answered_count,
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": exact_match_count / total_count if total_count else 0.0,
        "per_subject": per_subject,
        "failure_count": failure_count,
        "fallback_like_count": fallback_like_count,
        "average_answer_length": average_answer_length,
        "average_reasoning_process_length": average_reasoning_length,
    }


def compare_answers(prediction: Any, gold: Any, tolerance: float = 1e-6) -> tuple[bool, bool]:
    pred_text = _normalize_answer_text(prediction)
    gold_text = _normalize_answer_text(gold)
    if pred_text == gold_text:
        return True, _is_pure_numeric(pred_text) and _is_pure_numeric(gold_text)

    pred_num = _maybe_parse_number(pred_text)
    gold_num = _maybe_parse_number(gold_text)
    if pred_num is not None and gold_num is not None:
        matched = abs(pred_num - gold_num) <= tolerance
        return matched, matched
    return False, False


def _ensure_questions(items: list[Any], path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"dev set item {index} in {path} must be an object")
        questions.append(item)
    return questions


def _is_fallback_like(prediction: dict[str, Any]) -> bool:
    text = f"{prediction.get('prediction', '')} {prediction.get('reasoning_process', '')}"
    return any(hint in text for hint in FALLBACK_HINTS)


def _write_json(path: str | Path, payload: Any) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _average(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_answer_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _is_pure_numeric(text: str) -> bool:
    return _maybe_parse_number(text) is not None


def _maybe_parse_number(text: str) -> float | None:
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return None
    return None
