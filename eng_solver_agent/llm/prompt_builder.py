"""Two-stage prompt builders for analysis and drafting.

The templates stay explicit and low-noise so they can be swapped between
vendors. The analyze stage is JSON-only. The draft stage expects tool outputs
and produces competition-style reasoning plus a final answer.
"""

from __future__ import annotations

import json
from typing import Any


ANALYZE_REQUIRED_FIELDS = (
    "subject",
    "topic",
    "knowns",
    "unknowns",
    "equations_or_theorems",
    "should_use_tool",
    "target_form",
    "possible_traps",
)
DRAFT_REQUIRED_FIELDS = ("reasoning_process", "answer")

SUBJECT_TEMPLATES = {
    "physics": {
        "label": "physics",
        "focus": "kinematics, Newton's laws, work-energy, momentum, units, and constraints",
        "tool_hint": "Use the algebra or physics tool when equations need solving or checking.",
        "draft_hint": "Show knowns first, then unknowns, then formulas, substitution, and the final value with units.",
    },
    "circuits": {
        "label": "circuits",
        "focus": "series/parallel resistance, KCL, KVL, node analysis, mesh analysis, RC/RL transients",
        "tool_hint": "Use the circuit or algebra tool for structured netlists or simultaneous equations.",
        "draft_hint": "Show known circuit quantities first, then equations, node/mesh setup, substitution, and the final answer with units.",
    },
    "linalg": {
        "label": "linalg",
        "focus": "linear systems, matrix inverse, determinant, rank, eigenvalues, eigenvectors, simplification",
        "tool_hint": "Use the algebra tool when exact matrix or symbolic manipulation is needed.",
        "draft_hint": "Show the matrix facts first, then the theorem or formula, substitution or elimination, and the final result.",
    },
    "calculus": {
        "label": "calculus",
        "focus": "limits, derivatives, integrals, critical points, and Taylor expansion",
        "tool_hint": "Use the calculus tool when direct symbolic differentiation or integration is needed.",
        "draft_hint": "Show the function and target first, then the calculus rule, derivation, and the final value or expression.",
    },
}


def build_analyze_prompt(
    question: Any,
    subject: str | None = None,
    retrieval_context: Any | None = None,
) -> str:
    question_text = _question_text(question)
    subject_key = _normalize_subject(subject or _question_subject(question))
    template = SUBJECT_TEMPLATES.get(subject_key, SUBJECT_TEMPLATES["physics"])
    payload = {
        "question_id": _question_id(question),
        "question": question_text,
        "subject_hint": subject_key,
    }
    return "\n".join(
        [
            f"你正在分析一道{template['label']}题目。",
            "请输出严格JSON格式。不要使用markdown。不要添加额外文本。不要使用代码块标记。",
            "必须包含的JSON字段:",
            _format_required_fields(ANALYZE_REQUIRED_FIELDS),
            "字段规则:",
            "- subject: physics, circuits, linalg, calculus 之一。",
            "- topic: 简短的主题名称。",
            "- knowns 和 unknowns: 短字符串数组。",
            "- equations_or_theorems: 需要使用的公式、定律或定理数组。",
            "- should_use_tool: boolean。",
            "- target_form: 描述期望的求解形式。",
            "- possible_traps: 短字符串数组。",
            "学科重点:",
            template["focus"],
            "工具提示:",
            template["tool_hint"],
            *_retrieval_lines(retrieval_context),
            "题目JSON:",
            json.dumps(payload, ensure_ascii=False),
        ]
    )


def build_draft_prompt(
    question: Any,
    analysis: Any,
    tool_results: Any | None = None,
    subject: str | None = None,
    retrieval_context: Any | None = None,
) -> str:
    question_text = _question_text(question)
    subject_key = _normalize_subject(subject or _analysis_subject(analysis) or _question_subject(question))
    template = SUBJECT_TEMPLATES.get(subject_key, SUBJECT_TEMPLATES["physics"])
    payload = {
        "question_id": _question_id(question),
        "question": question_text,
        "analysis": _to_dict(analysis),
        "tool_results": _to_dict(tool_results) if tool_results is not None else [],
    }
    return "\n".join(
        [
            f"你正在为一道{template['label']}题目撰写最终答案。",
            "请输出严格JSON格式。不要使用markdown。不要添加额外文本。不要使用代码块标记。",
            "必须包含的JSON字段:",
            _format_required_fields(DRAFT_REQUIRED_FIELDS),
            "撰写规则:",
            "- 从已知条件和未知量开始。",
            "- 说明使用的公式、定理或控制方程。",
            "- 展示代入和推导步骤。",
            "- 以结果和任何必要的单位或条件结束。",
            template["draft_hint"],
            *_retrieval_lines(retrieval_context),
            "题目与分析结果JSON:",
            json.dumps(payload, ensure_ascii=False),
        ]
    )


def build_analyze_messages(
    question: Any,
    subject: str | None = None,
    retrieval_context: Any | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "请返回严格JSON格式。"},
        {"role": "user", "content": build_analyze_prompt(question, subject=subject, retrieval_context=retrieval_context)},
    ]


def build_draft_messages(
    question: Any,
    analysis: Any,
    tool_results: Any | None = None,
    subject: str | None = None,
    retrieval_context: Any | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "请返回严格JSON格式。"},
        {
            "role": "user",
            "content": build_draft_prompt(
                question,
                analysis,
                tool_results=tool_results,
                subject=subject,
                retrieval_context=retrieval_context,
            ),
        },
    ]


def build_solver_prompt(question: Any, route: str) -> str:
    return build_analyze_prompt(question, subject=route)


def _format_required_fields(fields: tuple[str, ...]) -> str:
    return "\n".join(f"- {field}" for field in fields)


def _question_text(question: Any) -> str:
    if isinstance(question, dict):
        return str(question.get("question") or question.get("prompt") or "")
    return str(getattr(question, "question", "") or getattr(question, "prompt", "") or "")


def _question_id(question: Any) -> Any:
    if isinstance(question, dict):
        return question.get("question_id") or question.get("id") or "unknown"
    return getattr(question, "question_id", getattr(question, "id", "unknown"))


def _question_subject(question: Any) -> str | None:
    if isinstance(question, dict):
        return question.get("subject")
    return getattr(question, "subject", None)


def _analysis_subject(analysis: Any) -> str | None:
    data = _to_dict(analysis)
    subject = data.get("subject")
    return str(subject) if subject else None


def _normalize_subject(subject: str) -> str:
    normalized = str(subject).strip().lower()
    if normalized in {"physics", "circuits", "linalg", "calculus"}:
        return normalized
    if normalized in {"linear_algebra", "matrix", "algebra"}:
        return "linalg"
    if normalized in {"circuit", "circuit_analysis"}:
        return "circuits"
    if normalized in {"calc", "analysis"}:
        return "calculus"
    return "physics"


def _to_dict(value: Any) -> dict[str, Any] | list[Any]:
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, (dict, list)):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, (dict, list)):
            return dumped
    return {"value": str(value)}


def _retrieval_lines(retrieval_context: Any | None) -> list[str]:
    if retrieval_context is None:
        return []
    data = _to_dict(retrieval_context)
    if not isinstance(data, dict):
        return []
    formula_cards = data.get("formula_cards") or []
    solved_examples = data.get("solved_examples") or []
    if not formula_cards and not solved_examples:
        return []
    lines = ["Retrieved context:"]
    for card in list(formula_cards)[:2]:
        if not isinstance(card, dict):
            continue
        lines.append(
            "- formula card: "
            + ", ".join(
                filter(
                    None,
                    [
                        str(card.get("topic", "")).strip(),
                        str(card.get("formula", "")).strip(),
                        "conditions: " + "; ".join(card.get("conditions", [])[:2]) if card.get("conditions") else "",
                    ],
                )
            )
        )
    for example in list(solved_examples)[:1]:
        if not isinstance(example, dict):
            continue
        lines.append(
            "- solved example: "
            + ", ".join(
                filter(
                    None,
                    [
                        str(example.get("topic", "")).strip(),
                        str(example.get("question", "")).strip(),
                        str(example.get("answer", "")).strip(),
                    ],
                )
            )
        )
    return lines
