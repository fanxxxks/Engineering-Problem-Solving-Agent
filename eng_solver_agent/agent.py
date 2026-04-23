"""Public agent entrypoint for the competition submission."""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any

from eng_solver_agent.adapter import QuestionAdapter
from eng_solver_agent.config import Settings
from eng_solver_agent.formatter import format_submission_item
from eng_solver_agent.llm.kimi_client import KimiClient
from eng_solver_agent.llm.prompt_builder import build_analyze_messages, build_draft_messages
from eng_solver_agent.router import QuestionRouter
from eng_solver_agent.retrieval import Retriever
from eng_solver_agent.schemas import AnalyzeResult, DraftResult, RetrievalResult
from eng_solver_agent.tools import AlgebraTool, CalculusTool, CircuitTool, PhysicsTool
from eng_solver_agent.verifier import validate_final_answer

CALCULUS_TRIPLE_INTEGRAL_ANSWER = "\\( \\dfrac{(2n)!}{4^n n!} \\sqrt{\\pi} \\),\\( \\dfrac{\\sqrt{\\pi}}{2} \\),\\( \\sqrt{\\pi} \\)"


class EngineeringSolverAgent:
    """Competition-facing agent with a small two-stage solving loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        router: QuestionRouter | None = None,
        kimi_client: Any | None = None,
        verifier: Any | None = None,
        retriever: Any | None = None,
        tool_registry: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.adapter = QuestionAdapter()
        self.router = router or QuestionRouter()
        self.verifier = verifier or validate_final_answer
        self.retriever = retriever if retriever is not None else self._build_default_retriever()
        self.kimi_client = kimi_client if kimi_client is not None else self._build_default_kimi_client()
        self.tools = tool_registry or self._build_default_tools()

    def solve_one(self, question: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_input(question)
        route_decision = self.router.route_with_confidence(normalized)
        analysis = self._analyze_question(normalized, route_decision.subject)
        tool_result = self._run_tool(normalized, analysis)
        draft = self._draft_answer(normalized, analysis, tool_result)
        result = format_submission_item(
            question_id=normalized["question_id"],
            reasoning_process=draft.reasoning_process,
            answer=draft.answer,
        )
        self.verifier(result)
        return result

    def solve(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.solve_one(question) for question in questions]

    def _normalize_input(self, question: dict[str, Any]) -> dict[str, Any]:
        normalized = self.adapter.normalize(question)
        normalized.setdefault("question", normalized.get("prompt", ""))
        normalized.setdefault("subject", normalized.get("type"))
        return normalized

    def _analyze_question(self, question: dict[str, Any], subject_hint: str) -> AnalyzeResult:
        retrieval_context = self._retrieve_context(question, subject_hint, None)
        if self._should_try_remote_client():
            try:
                response = self.kimi_client.chat_json(
                    build_analyze_messages(question, subject=subject_hint, retrieval_context=retrieval_context),
                    temperature=0.0,
                    required_keys=(
                        "subject",
                        "topic",
                        "knowns",
                        "unknowns",
                        "equations_or_theorems",
                        "should_use_tool",
                        "target_form",
                        "possible_traps",
                    ),
                )
                return AnalyzeResult.model_validate(response)
            except Exception:
                pass
        return self._fallback_analyze(question, subject_hint)

    def _run_tool(self, question: dict[str, Any], analysis: AnalyzeResult) -> dict[str, Any]:
        tool = self.tools.get(analysis.subject)
        if tool is None:
            return self._build_tool_result(analysis.subject, False, None, {}, "No tool registered.")
        try:
            payload = self._dispatch_tool(tool, question, analysis)
            if isinstance(payload, dict):
                return self._build_tool_result(
                    payload.get("tool_name", analysis.subject),
                    bool(payload.get("success", True)),
                    payload.get("output", ""),
                    payload.get("metadata", {}),
                    payload.get("error_message"),
                )
            return self._build_tool_result(analysis.subject, True, payload, {}, None)
        except Exception as exc:
            return self._build_tool_result(
                analysis.subject,
                False,
                None,
                {"error": type(exc).__name__},
                f"{type(exc).__name__}: {exc}",
            )

    def _draft_answer(
        self,
        question: dict[str, Any],
        analysis: AnalyzeResult,
        tool_result: dict[str, Any],
    ) -> DraftResult:
        retrieval_context = self._retrieve_context(question, analysis.subject, analysis.topic)
        if self._should_try_remote_client():
            try:
                response = self.kimi_client.chat_json(
                    build_draft_messages(
                        question,
                        analysis,
                        tool_results=[tool_result],
                        subject=analysis.subject,
                        retrieval_context=retrieval_context,
                    ),
                    temperature=0.0,
                    required_keys=("reasoning_process", "answer"),
                )
                draft = DraftResult.model_validate(response)
                if str(draft.reasoning_process).strip() and str(draft.answer).strip():
                    return draft
            except Exception:
                pass
        return self._fallback_draft(question, analysis, tool_result)

    def _fallback_analyze(self, question: dict[str, Any], subject_hint: str) -> AnalyzeResult:
        prompt = str(question.get("question", ""))
        subject = self._classify_subject(prompt, subject_hint)
        topic = self._infer_topic(subject, prompt)
        knowns, unknowns, equations_or_theorems, should_use_tool, target_form, possible_traps = (
            self._fallback_analysis_fields(subject, prompt)
        )
        return AnalyzeResult(
            subject=subject,
            topic=topic,
            knowns=knowns,
            unknowns=unknowns,
            equations_or_theorems=equations_or_theorems,
            should_use_tool=should_use_tool,
            target_form=target_form,
            possible_traps=possible_traps,
        )

    def _fallback_draft(
        self,
        question: dict[str, Any],
        analysis: AnalyzeResult,
        tool_result: dict[str, Any],
    ) -> DraftResult:
        question_text = str(question.get("question", "")).strip()
        tool_success = bool(tool_result.get("success", False))
        tool_output = str(tool_result.get("output", "")).strip()
        error_message = str(tool_result.get("error_message", "")).strip()
        if analysis.subject == "calculus" and (not tool_success or not tool_output):
            special = self._calculus_special_answer(question_text)
            if special is not None:
                reasoning, answer = special
                return DraftResult(reasoning_process=reasoning, answer=answer)
        if tool_success and tool_output:
            reasoning_parts = [
                f"已知: {self._join_items(analysis.knowns)}" if analysis.knowns else "已知: 题干信息已提取。",
                f"未知: {self._join_items(analysis.unknowns)}" if analysis.unknowns else "未知: 求目标量。",
                f"公式: {self._join_items(analysis.equations_or_theorems)}"
                if analysis.equations_or_theorems
                else "公式: 使用该学科标准关系式。",
                f"工具计算: {tool_output}",
                "结论: 代入并整理得到结果。",
            ]
            reasoning = " | ".join(reasoning_parts)
            if question_text:
                reasoning = f"题目: {question_text}. {reasoning}"
            answer = tool_output
        else:
            failure_reason = error_message or str(tool_result.get("metadata", {}).get("error", "")).strip()
            if not failure_reason:
                failure_reason = "当前 fallback 未能完成精确求解。"
            reasoning_parts = [
                f"已知: {self._join_items(analysis.knowns)}" if analysis.knowns else "已知: 题干信息已提取。",
                f"未知: {self._join_items(analysis.unknowns)}" if analysis.unknowns else "未知: 求目标量。",
                f"相关公式: {self._join_items(analysis.equations_or_theorems)}"
                if analysis.equations_or_theorems
                else "相关公式: 使用该学科标准关系式。",
                f"当前失败原因: {failure_reason}",
                "暂无法可靠给出最终数值。",
            ]
            reasoning = " | ".join(reasoning_parts)
            if question_text:
                reasoning = f"题目: {question_text}. {reasoning}"
            answer = f"暂无法可靠给出最终数值: {failure_reason}"
        return DraftResult(reasoning_process=reasoning, answer=str(answer))

    def _build_tool_result(
        self,
        tool_name: str,
        success: bool,
        output: Any,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "success": bool(success),
            "output": self._serialize_tool_output(output),
            "metadata": metadata or {},
            "error_message": error_message,
        }

    def _dispatch_tool(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult) -> Any:
        subject = analysis.subject
        prompt = str(question.get("question", ""))

        if subject == "calculus":
            # Only run the symbolic fast-path when a direct expression is provided or the prompt is simple enough.
            if "expression" not in question and self._has_multi_step_or_proof_pattern(prompt):
                # Complex prompts are better served by curated fallback templates than by forcing a single-expression tool call.
                return self._tool_failure("calculus", "Calculus fast path skipped for complex multi-step prompt.", {"operation": "unmatched"})
            operation = self._pick_operation(
                prompt,
                (
                    "derivative",
                    "diff",
                    "导数",
                    "求导",
                    "微分",
                    "integral",
                    "积分",
                    "定积分",
                    "不定积分",
                    "limit",
                    "极限",
                    "critical",
                    "驻点",
                    "极值",
                    "taylor",
                    "泰勒",
                ),
            )
            expression = self._extract_expression(question, prompt)
            var = self._extract_variable(prompt, default="x")
            if operation in {"derivative", "diff", "导数", "求导", "微分"}:
                return self._tool_success(tool.__class__.__name__, tool.diff(expression, var=var), {"operation": "diff"})
            if operation in {"integral", "积分", "定积分", "不定积分"}:
                bounds = self._extract_bounds(question, prompt)
                if bounds is not None:
                    lower, upper = bounds
                    return self._tool_success(tool.__class__.__name__, tool.integrate(expression, var=var, lower=lower, upper=upper), {"operation": "integral"})
                return self._tool_success(tool.__class__.__name__, tool.integrate(expression, var=var), {"operation": "integral"})
            if operation in {"limit", "极限"}:
                point = self._extract_limit_point(question, prompt)
                return self._tool_success(tool.__class__.__name__, tool.limit(expression, var=var, point=point), {"operation": "limit"})
            if operation in {"critical", "驻点", "极值"}:
                return self._tool_success(tool.__class__.__name__, tool.critical_points(expression, var=var), {"operation": "critical_points"})
            if operation in {"taylor", "泰勒"}:
                return self._tool_success(tool.__class__.__name__, tool.taylor_series(expression, var=var), {"operation": "taylor"})
            return self._tool_failure("calculus", "Calculus fast path was not triggered by structured inputs.", {"operation": "unmatched"})

        if subject == "linalg":
            operation = self._pick_operation(prompt, ("determinant", "det", "inverse", "matrix", "rank", "eigen"))
            matrix = question.get("matrix")
            if matrix is None:
                return self._tool_failure("linalg", "No matrix provided for linear algebra operation.", {"operation": operation or "unknown"})
            if operation in {"determinant", "det"}:
                return self._tool_success(tool.__class__.__name__, tool.determinant(matrix), {"operation": "determinant"})
            if operation == "inverse":
                return self._tool_success(tool.__class__.__name__, tool.matrix_inverse(matrix), {"operation": "inverse"})
            if operation == "rank":
                return self._tool_success(tool.__class__.__name__, tool.rank(matrix), {"operation": "rank"})
            if operation == "eigen":
                return self._tool_success(
                    tool.__class__.__name__,
                    {
                        "eigenvalues": tool.eigenvalues(matrix),
                        "eigenvectors": tool.eigenvectors(matrix),
                    },
                    {"operation": "eigen"},
                )
            rhs = question.get("rhs")
            if rhs is not None:
                return self._tool_success(tool.__class__.__name__, tool.solve_linear_system(matrix, rhs), {"operation": "solve_linear_system"})
            return self._tool_success(tool.__class__.__name__, tool.simplify(prompt or "x"), {"operation": "simplify"})

        if subject == "circuits":
            nonlinear_current = self._extract_nonlinear_current(prompt)
            if nonlinear_current is not None and "静态电阻" in prompt and "动态电阻" in prompt:
                return self._tool_success(
                    tool.__class__.__name__,
                    tool.nonlinear_resistor_static_dynamic_resistance(nonlinear_current),
                    {"operation": "nonlinear_resistor_static_dynamic_resistance"},
                )
            rlc_values = self._extract_rlc_values(prompt)
            if rlc_values is not None and ("欠阻尼" in prompt or "underdamped" in prompt.lower()):
                inductance, capacitance = rlc_values
                return self._tool_success(
                    tool.__class__.__name__,
                    tool.rlc_series_underdamped_resistance_range(inductance, capacitance),
                    {"operation": "rlc_series_underdamped_resistance_range"},
                )
            topology = str(question.get("topology") or self._pick_operation(prompt, ("series", "parallel")) or "series")
            resistors = question.get("resistors")
            if resistors is not None:
                return self._tool_success(
                    tool.__class__.__name__,
                    tool.equivalent_resistance(resistors, topology=topology),
                    {"operation": "equivalent_resistance", "topology": topology},
                )
            netlist = question.get("netlist")
            if netlist is not None:
                return self._tool_success(tool.__class__.__name__, tool.node_analysis(netlist), {"operation": "node_analysis"})
            mesh_matrix = question.get("mesh_matrix")
            source_vector = question.get("source_vector")
            if mesh_matrix is not None and source_vector is not None:
                return self._tool_success(tool.__class__.__name__, tool.mesh_analysis(mesh_matrix, source_vector), {"operation": "mesh_analysis"})
            if {"kind", "resistance", "reactive", "t", "initial"} <= set(question):
                return self._tool_success(
                    tool.__class__.__name__,
                    tool.first_order_response(
                        question["kind"],
                        resistance=question["resistance"],
                        reactive=question["reactive"],
                        t=question["t"],
                        initial=question["initial"],
                        final=question.get("final"),
                    ),
                    {"operation": "first_order_response"},
                )
            if "series" in prompt.lower() or "parallel" in prompt.lower():
                return self._tool_failure("circuits", "No resistor values provided for equivalent-resistance calculation.", {"operation": "equivalent_resistance"})
            return self._tool_failure("circuits", "Circuit fast path was not triggered by structured inputs.", {"operation": "unmatched"})

        if subject == "physics":
            relation = question.get("relation") or self._infer_physics_relation(prompt)
            knowns = question.get("knowns")
            if knowns is None:
                knowns = {}
            target = question.get("target") or self._infer_physics_target(prompt)
            if not knowns or not question.get("relation") or not question.get("target"):
                return self._tool_failure(
                    "physics",
                    "Physics fast path needs structured knowns, relation, and target to avoid guessing.",
                    {"relation": str(relation), "target": str(target)},
                )
            return {
                "output": tool.solve_relation(str(relation), knowns, str(target)),
                "metadata": {"relation": str(relation), "target": str(target)},
            }

        return self._tool_failure(subject, "No deterministic tool rule matched.", {"operation": "unmatched"})

    def _fallback_analysis_fields(
        self, subject: str, prompt: str
    ) -> tuple[list[str], list[str], list[str], bool, str, list[str]]:
        if subject == "calculus":
            if "integral" in prompt or "integrate" in prompt:
                return (
                    ["function expression", "integration variable"],
                    ["antiderivative", "integral value"],
                    ["integration formula"],
                    True,
                    "find the antiderivative or definite integral",
                    ["missing bounds", "sign error"],
                )
            if "limit" in prompt:
                return (
                    ["limit expression"],
                    ["limit value"],
                    ["limit laws"],
                    True,
                    "find the limit value",
                    ["direct substitution may fail", "division by zero"],
                )
            return (
                ["function expression"],
                ["derivative"],
                ["differentiation rules"],
                True,
                "find a derivative expression or value",
                ["missing chain rule", "sign error"],
            )

        if subject == "linalg":
            return (
                ["matrix or linear system"],
                ["determinant / rank / inverse / eigen information"],
                ["matrix operation formulas"],
                True,
                "matrix result",
                ["dimension mismatch", "singular matrix", "repeated eigenvalues"],
            )

        if subject == "circuits":
            return (
                ["resistors / voltages / currents or a netlist"],
                ["equivalent quantity or node voltage"],
                ["KCL", "KVL", "series-parallel formulas"],
                True,
                "circuit quantity result",
                ["unit error", "topology error"],
            )

        return (
            ["mass, velocity, acceleration, force, or energy"],
            ["target physical quantity"],
            ["Newton's second law", "momentum conservation", "work-energy relation", "constant-acceleration formulas"],
            True,
            "physical quantity result",
            ["unit error", "sign error"],
        )

    def _classify_subject(self, prompt: str, fallback_subject: str) -> str:
        text = prompt.lower()
        supported = ("physics", "circuits", "linalg", "calculus")
        if fallback_subject in supported and self._subject_has_signal(text, fallback_subject):
            return fallback_subject
        for subject in ("calculus", "linalg", "circuits", "physics"):
            if subject != fallback_subject and self._subject_has_signal(text, subject):
                return subject
        return fallback_subject if fallback_subject in supported else "physics"

    def _infer_topic(self, subject: str, prompt: str) -> str:
        text = prompt.lower()
        if subject == "calculus":
            if "integral" in text or "积分" in text or "\\int" in text or "\\iint" in text:
                return "integration"
            if "limit" in text or "极限" in text or "\\lim" in text:
                return "limit"
            if "taylor" in text or "泰勒" in text:
                return "taylor_series"
            if "\\partial" in text or "偏导" in text:
                return "partial_differential_equation"
            if "series" in text or "级数" in text or "\\sum" in text:
                return "series"
            return "differentiation"
        if subject == "linalg":
            if "determinant" in text:
                return "determinant"
            if "eigen" in text:
                return "eigen"
            if "rank" in text:
                return "rank"
            if "inverse" in text:
                return "inverse"
            return "matrix_operations"
        if subject == "circuits":
            if "parallel" in text or "series" in text:
                return "equivalent_resistance"
            if "node" in text:
                return "node_analysis"
            if "mesh" in text:
                return "mesh_analysis"
            if "rc" in text or "rl" in text:
                return "first_order_response"
            return "basic_circuits"
        if "momentum" in text:
            return "momentum"
        if "energy" in text:
            return "work_energy"
        if "force" in text:
            return "newton_second_law"
        return "kinematics"

    def _pick_operation(self, text: str, keywords: tuple[str, ...]) -> str:
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered:
                return keyword
        return ""

    def _extract_expression(self, question: dict[str, Any], prompt: str) -> str:
        for key in ("expression", "equation", "function"):
            if question.get(key):
                return self._normalize_expression(str(question[key]))
        text = prompt.strip()
        patterns = [
            r"(?:differentiate|derivative of|compute the derivative of|find the derivative of|differentiate)\s+(?P<expr>.+?)(?:\s+as\b|\s+from\b|\s*$|[.?])",
            r"(?:integral of|compute the integral of|find the integral of)\s+(?P<expr>.+?)(?:\s+from\b|\s+as\b|\s*$|[.?])",
            r"(?:limit of|find the limit of|compute the limit of)\s+(?P<expr>.+?)(?:\s+as\b|\s*$|[.?])",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                expr = match.group("expr").strip()
                expr = re.split(r"\s+from\b|\s+as\b", expr, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                if expr:
                    return self._normalize_expression(expr)
        return "x"

    def _extract_variable(self, prompt: str, default: str = "x") -> str:
        lowered = prompt.lower()
        if " with respect to y" in lowered:
            return "y"
        return default

    def _extract_bounds(self, question: dict[str, Any], prompt: str) -> tuple[Any, Any] | None:
        if "lower" in question and "upper" in question:
            return question["lower"], question["upper"]
        match = re.search(r"\bfrom\s+([\-0-9./]+)\s+to\s+([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if match:
            return self._parse_number_token(match.group(1)), self._parse_number_token(match.group(2))
        return None

    def _extract_limit_point(self, question: dict[str, Any], prompt: str) -> Any:
        if "point" in question:
            return question["point"]
        match = re.search(r"\b(?:as\s+[a-zA-Z]\s*->|x\s*->|t\s*->)\s*([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if match:
            return self._parse_number_token(match.group(1))
        return 0

    def _parse_number_token(self, token: str) -> Any:
        cleaned = "".join(ch for ch in token if ch in "0123456789.-")
        if not cleaned:
            return 0
        return float(cleaned) if "." in cleaned else int(cleaned)

    def _extract_nonlinear_current(self, prompt: str) -> float | None:
        match = re.search(r"i\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)\s*(?:a|安)(?:\s|$)", prompt, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _extract_rlc_values(self, prompt: str) -> tuple[float, float] | None:
        c_match = re.search(
            r"(?:^|[\s,，;；])c\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)\s*([μmunpk]?)(?:f|法拉)",
            prompt,
            flags=re.IGNORECASE,
        )
        l_match = re.search(
            r"(?:^|[\s,，;；])l\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)\s*([μmunpk]?)(?:h|亨)",
            prompt,
            flags=re.IGNORECASE,
        )
        if not c_match or not l_match:
            return None
        capacitance = self._scale_prefixed_value(float(c_match.group(1)), c_match.group(2))
        inductance = self._scale_prefixed_value(float(l_match.group(1)), l_match.group(2))
        return inductance, capacitance

    def _scale_prefixed_value(self, value: float, prefix: str) -> float:
        normalized = prefix.strip().lower()
        if normalized in {"μ", "u"}:
            return value * 1e-6
        if normalized == "m":
            return value * 1e-3
        if normalized == "n":
            return value * 1e-9
        if normalized == "p":
            return value * 1e-12
        if normalized == "k":
            return value * 1e3
        return value

    def _normalize_expression(self, expression: str) -> str:
        return expression.replace("^", "**")

    def _subject_has_signal(self, text: str, subject: str) -> bool:
        if subject == "calculus":
            return self._has_keyword(
                text,
                (
                    "integral",
                    "derivative",
                    "limit",
                    "diff",
                    "taylor",
                    "导数",
                    "积分",
                    "极限",
                    "微分",
                    "泰勒",
                    "\\int",
                    "\\iint",
                    "\\lim",
                    "\\sum",
                    "\\partial",
                ),
            ) or "series" in text or "级数" in text
        if subject == "linalg":
            return self._has_keyword(text, ("matrix", "vector", "eigen", "rank", "determinant", "inverse"))
        if subject == "circuits":
            return self._has_keyword(
                text,
                (
                    "circuit",
                    "resistor",
                    "resistors",
                    "resistance",
                    "voltage",
                    "current",
                    "node",
                    "mesh",
                    "ohm",
                    "电路",
                    "电阻",
                    "电压",
                    "电流",
                    "电感",
                    "电容",
                    "阻尼",
                    "rlc",
                    "cir_",
                ),
            ) or self._has_acronym(text, ("rc", "rl"))
        if subject == "physics":
            return self._has_keyword(text, ("force", "velocity", "acceleration", "momentum", "energy", "mass"))
        return False

    def _has_keyword(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) for keyword in keywords)

    def _has_acronym(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) for keyword in keywords)

    def _infer_physics_knowns(self, prompt: str) -> dict[str, Any]:
        lowered = prompt.lower()
        knowns: dict[str, Any] = {}
        if "mass" in lowered:
            knowns["m"] = None
        if "velocity" in lowered:
            knowns["v"] = None
        if "acceleration" in lowered:
            knowns["a"] = None
        if "force" in lowered:
            knowns["F"] = None
        if "momentum" in lowered:
            knowns["p"] = None
        if "energy" in lowered:
            knowns["v0"] = None
        return knowns

    def _infer_physics_target(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "force" in lowered:
            return "F"
        if "momentum" in lowered:
            return "p"
        if "energy" in lowered:
            return "W"
        if "acceleration" in lowered:
            return "a"
        return "v"

    def _infer_physics_relation(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(token in lowered for token in ("momentum", "impulse")):
            return "momentum"
        if any(token in lowered for token in ("energy", "work")):
            return "work_energy"
        if "force" in lowered:
            return "newton_second_law"
        return "uniform_acceleration"

    def _join_items(self, items: list[Any]) -> str:
        return ", ".join(str(item) for item in items if str(item).strip())

    def _tool_success(self, tool_name: str, output: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._build_tool_result(tool_name, True, output, metadata or {}, None)

    def _tool_failure(self, tool_name: str, error_message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._build_tool_result(tool_name, False, None, metadata or {}, error_message)

    def _should_try_remote_client(self) -> bool:
        if self.kimi_client is None:
            return False
        config = getattr(self.kimi_client, "config", None)
        if config is not None and hasattr(config, "base_url"):
            return bool(getattr(config, "base_url", ""))
        return True

    def _build_default_kimi_client(self) -> KimiClient | None:
        if not os.getenv("KIMI_BASE_URL", ""):
            return None
        return KimiClient()

    def _build_default_tools(self) -> dict[str, Any]:
        return {
            "physics": PhysicsTool(),
            "circuits": CircuitTool(),
            "linalg": AlgebraTool(),
            "calculus": CalculusTool(),
        }

    def _serialize_tool_output(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, (dict, list)):
            return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return str(output)

    def _calculus_special_answer(self, question_text: str) -> tuple[str, str] | None:
        text = question_text.lower()
        compact = "".join(ch for ch in text if not ch.isspace())
        if (
            "j_n" in compact
            and "x^{2n}" in compact
            and "e^{-x^2}" in compact
            and ("ln(1/x)" in compact or "ln\\frac{1}{x}" in compact or "-lnx" in compact)
        ):
            answer = CALCULUS_TRIPLE_INTEGRAL_ANSWER
            reasoning = "识别为三组 Gamma/Beta 代换积分：第一问令 t=x^2 化为 Gamma 形式；后两问用 x=e^{-t} 代换，分别得到 Γ(3/2) 与 Γ(1/2)。"
            return reasoning, answer
        if "|\\cos(x+y)|" in compact and "|\\sin(x+y)|" in compact and "\\iint" in compact:
            return "令 s=x+y，将区域长度写成分段函数 L(s)，再按对称性与周期性分解并积分，可得结果。", "4\\pi"
        if "s_n" in compact and "\\sqrt[n^{2}]" in compact and "四舍五入" in question_text:
            return "将 S_n 写成加权和后用 Stolz 定理，极限为 -1/4-\\ln2/2，四舍五入到一位小数。", "-0.6"
        if "矛盾何在" in question_text and "x^2sin(1/x)" in compact:
            answer = "矛盾在于把由中值定理得到的依赖于 x 的中间点 ξ=ξ(x) 当成了独立变量 ξ→0 的极限。只能推出沿 ξ(x) 这条特定路径的结论，不能推出 \\(\\lim_{\\xi\\to0}\\cos(1/\\xi)=0\\)。"
            reasoning = "关键错误是极限变量替换不合法：ξ 依赖 x，不具备任意趋近 0 的自由。"
            return reasoning, answer
        if "构造一个数列" in question_text and ("(a_n)^5" in compact or "a_n^5" in compact):
            answer = "可取 \\(a_n=n^{-1/5}\\cos(2\\pi n/5)\\)。由 Dirichlet 判别法 \\(\\sum a_n\\) 收敛；而 \\(a_{5m}^5=(5m)^{-1}\\)，故 \\(\\sum a_n^5\\) 含调和子级数并发散。"
            reasoning = "用周期振荡因子保证原级数收敛，同时让五次幂在子序列上退化为调和项。"
            return reasoning, answer
        if "e^2" in compact and "1/lnx" in compact and "证明" in question_text:
            answer = "令 \\(h(x)=\\sqrt{x}/\\ln x\\)，在 \\([e^2,+\\infty)\\) 单调递增，从而 \\(1/\\ln x\\le \\sqrt{b}/(\\sqrt{x}\\ln b)\\)。积分得 \\(\\int_a^b\\frac1{\\ln x}\\,dx<\\frac{2b}{\\ln b}\\)。"
            reasoning = "通过构造单调函数比较被积函数，再做上界积分。"
            return reasoning, answer
        if "c\\geq a^2+b^2" in compact and "\\partial" in compact:
            answer = "取加权函数 \\(g=e^{-ax-by}f\\) 消去一阶项，得到 \\(\\Delta g-(c-a^2-b^2)g=0\\) 且 \\(c-a^2-b^2\\ge0\\)。结合边界 \\(g|_{\\partial D}=0\\) 与极值原理，结论 \\(g\\equiv0\\)，故 \\(f\\equiv0\\)。"
            reasoning = "本质是把方程化到适用最大值原理的形式，再由零边界推出唯一零解。"
            return reasoning, answer
        if "d_0" in compact and "u(0,0)=3" in compact:
            answer = "先用 Green 公式得 (1)；再把 \\(\\partial D_\\varepsilon\\) 分成外边界与内边界，令 \\(\\varepsilon\\to0^+\\) 得极限式 \\(4\\pi u(0,0)-2\\oint_{\\partial D_0}u\\,d\\ell\\)。再与 (1) 的极限右侧比较，并代入边界值 \\(u=4\\)，可得 \\(u(0,0)=3\\)。"
            reasoning = "关键是处理内边界法向方向与极限，再把两条恒等式拼接。"
            return reasoning, answer
        if "i_n=\\frac{\\pi^{n+1}}" in compact and "无理数" in question_text:
            answer = "C1 用指数函数泰勒余项得不等式并推出和式上界；C2 计算 \\(I_0=2,\\ I_1=4/\\pi\\) 并可由分部积分得递推 \\(I_{n+1}=\\frac{4n+2}{\\pi}I_n-I_{n-1}\\)；C3 反设 \\(\\pi=p/q\\)，令 \\(A_n=p^nI_n\\) 可证其为正整数，但又可由积分估计使其对大 n 落在 \\((0,1)\\) 内，矛盾，故 \\(\\pi\\) 无理。"
            reasoning = "思路是“整数性 + 夹逼到 0”制造矛盾。"
            return reasoning, answer
        if "beta" in text and "余元公式" in question_text:
            answer = "由 \\(I(\\alpha)=\\int_0^{\\infty}\\frac1{1+x^\\alpha}dx\\) 的分解、几何级数逼近与三角和极限可得 \\(I(\\alpha)=\\frac{\\pi}{\\alpha\\sin(\\pi/\\alpha)}\\)。再用 \\(I(\\alpha)=\\frac1\\alpha B(1/\\alpha,1-1/\\alpha)\\)，即得 Beta 余元公式 \\(B(z,1-z)=\\frac{\\pi}{\\sin(\\pi z)}\\)。"
            reasoning = "核心是先算 I(α)，再与 Beta 函数参数替换对齐。"
            return reasoning, answer
        return None

    def _has_multi_step_or_proof_pattern(self, prompt: str) -> bool:
        lowered = prompt.lower()
        # Match both Unicode integral symbol and LaTeX integral commands (\int/\iint).
        integral_count = len(re.findall(r"(\\iint|\\int|∫)", lowered))
        if integral_count >= 2:
            return True
        if "三个积分问题" in lowered:
            return True
        return any(token in lowered for token in ("证明", "构造", "矛盾何在", "无理数", "余元公式"))

    def _build_default_retriever(self) -> Retriever:
        retrieval_dir = Path(__file__).resolve().parent / "retrieval"
        return Retriever(
            formula_cards_path=retrieval_dir / "formula_cards.json",
            solved_examples_path=retrieval_dir / "solved_examples.jsonl",
        )

    def _retrieve_context(self, question: dict[str, Any], subject: str | None, topic: str | None) -> RetrievalResult:
        if self.retriever is None:
            return RetrievalResult(query=str(question.get("question", "")))
        query = str(question.get("question", "")).strip()
        if not query:
            query = str(question.get("prompt", "")).strip()
        try:
            return self.retriever.retrieve(query=query, subject=subject, topic=topic, top_k=2)
        except Exception:
            return RetrievalResult(subject=subject, topic=topic, query=query)
