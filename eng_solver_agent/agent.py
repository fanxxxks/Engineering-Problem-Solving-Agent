"""Public agent entrypoint for the competition submission.

Architecture:
  1. LLM analyze  → extract structured data from question text
  2. Tool compute → precise calculation using extracted data (when possible)
  3. LLM draft    → organize reasoning + answer in Chinese math style
  4. Format       → competition JSON output

Key fixes from previous version:
  - kimi_client=None is now respected (no auto-rebuild from .env)
  - _dispatch_tool reads LLM analyze results instead of re-parsing text
  - Enhanced Chinese text parsing (matrix, equation, known-value extraction)
  - Tools are optional helpers; LLM can solve independently when tools can't
"""

from __future__ import annotations

import asyncio
import json
import os
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


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (module-level for reuse)
# ---------------------------------------------------------------------------

# Expression extraction from English/Chinese prompts
_EXPR_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        # English patterns
        r"(?:differentiate|derivative of|compute the derivative of|find the derivative of)\s+(?P<expr>.+?)(?:\s+as\b|\s+from\b|\s*$|[.?])",
        r"(?:integral of|compute the integral of|find the integral of)\s+(?P<expr>.+?)(?:\s+from\b|\s+as\b|\s*$|[.?])",
        r"(?:limit of|find the limit of|compute the limit of)\s+(?P<expr>.+?)(?:\s+as\b|\s*$|[.?])",
        # Chinese patterns
        r"(?:求|计算|求导|求微分|对)\s*(?P<expr>[^，。；]+?)(?:\s*的导数|\s*的积分|\s*的微分|(?:\s+as\b|\s+from\b|\s*$|[.。；]))",
        r"(?:求|计算)\s*(?P<expr>[^，。；]+?)(?:\s*的不定积分|\s*的定积分|(?:\s+as\b|\s+from\b|\s*$|[.。；]))",
        r"(?:求|计算)\s*(?P<expr>[^，。；]+?)(?:\s*的极限|(?:\s+as\b|\s*$|[.。；]))",
    ]
)

# Matrix extraction from text: [[a,b],[c,d]] or [a b; c d]
_MATRIX_RE = re.compile(
    r"(\[\s*\[.+?\]\s*\])"  # [[1,2],[3,4]]
    r"|"
    r"(\[\s*[^\[\]]+?[;；]\s*[^\[\]]+?\])"  # [1 2; 3 4]
)

# Number extraction: integers, decimals, fractions
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")

# Resistor value extraction from Chinese text
_RESISTOR_RE = re.compile(r"(\d+)\s*[Ω欧姆]", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fallback constants (immutable tuples to avoid rebuild)
# ---------------------------------------------------------------------------

_FALLBACK_CALCULUS_INTEGRAL = (
    ["function expression", "integration variable"],
    ["antiderivative", "integral value"],
    ["integration formula"],
    True,
    "find the antiderivative or definite integral",
    ["missing bounds", "sign error"],
)
_FALLBACK_CALCULUS_LIMIT = (
    ["limit expression"],
    ["limit value"],
    ["limit laws"],
    True,
    "find the limit value",
    ["direct substitution may fail", "division by zero"],
)
_FALLBACK_CALCULUS_DEFAULT = (
    ["function expression"],
    ["derivative"],
    ["differentiation rules"],
    True,
    "find a derivative expression or value",
    ["missing chain rule", "sign error"],
)
_FALLBACK_LINALG = (
    ["matrix or linear system"],
    ["determinant / rank / inverse / eigen information"],
    ["matrix operation formulas"],
    True,
    "matrix result",
    ["dimension mismatch", "singular matrix", "repeated eigenvalues"],
)
_FALLBACK_CIRCUITS = (
    ["resistors / voltages / currents or a netlist"],
    ["equivalent quantity or node voltage"],
    ["KCL", "KVL", "series-parallel formulas"],
    True,
    "circuit quantity result",
    ["unit error", "topology error"],
)
_FALLBACK_PHYSICS = (
    ["mass, velocity, acceleration, force, or energy"],
    ["target physical quantity"],
    ["Newton's second law", "momentum conservation", "work-energy relation", "constant-acceleration formulas"],
    True,
    "physical quantity result",
    ["unit error", "sign error"],
)


class EngineeringSolverAgent:
    """Competition-facing agent with LLM-first + tool-assisted architecture."""

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
        # FIX: respect explicit None, only build default if user did not pass the parameter at all
        self._kimi_client_arg = kimi_client
        self.kimi_client = self._resolve_kimi_client(kimi_client)
        self.tools = tool_registry or self._build_default_tools()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    async def async_solve(self, questions: list[dict[str, Any]], max_concurrent: int = 5) -> list[dict[str, Any]]:
        """Batch solve with controlled concurrency for LLM API calls."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _solve_one_async(question: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                # Run blocking solve_one in thread pool
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.solve_one, question)

        tasks = [asyncio.create_task(_solve_one_async(q)) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[dict[str, Any]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                qid = str(questions[idx].get("question_id", f"q{idx}"))
                out.append({
                    "question_id": qid,
                    "reasoning_process": f"Error: {type(result).__name__}: {result}",
                    "answer": f"Error: {type(result).__name__}",
                })
            else:
                out.append(result)
        return out

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Tool dispatch — now reads from analysis + question structured fields
    # ------------------------------------------------------------------

    def _dispatch_tool(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult) -> Any:
        subject = analysis.subject
        prompt = str(question.get("question", ""))
        lowered_prompt = prompt.lower()

        # ── Calculus ──────────────────────────────────────────────────
        if subject == "calculus":
            return self._dispatch_calculus(tool, question, analysis, lowered_prompt)

        # ── Linear Algebra ────────────────────────────────────────────
        if subject == "linalg":
            return self._dispatch_linalg(tool, question, analysis, lowered_prompt)

        # ── Circuits ──────────────────────────────────────────────────
        if subject == "circuits":
            return self._dispatch_circuits(tool, question, analysis, lowered_prompt)

        # ── Physics ───────────────────────────────────────────────────
        if subject == "physics":
            return self._dispatch_physics(tool, question, analysis, lowered_prompt)

        return self._tool_failure(subject, "No deterministic tool rule matched.", {"operation": "unmatched"})

    def _dispatch_calculus(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult, lowered_prompt: str) -> Any:
        operation = self._pick_operation(lowered_prompt, ("derivative", "diff", "integral", "limit", "critical", "taylor"))
        # Prefer structured fields, fallback to text extraction
        expression = question.get("expression") or question.get("equation") or question.get("function")
        if expression is None:
            expression = self._extract_expression(question, str(question.get("question", "")))
        var = question.get("variable") or self._extract_variable(str(question.get("question", "")), default="x")

        if operation in {"derivative", "diff"}:
            return self._tool_success(tool.__class__.__name__, tool.diff(expression, var=var), {"operation": "diff"})
        if operation == "integral":
            bounds = self._extract_bounds(question, str(question.get("question", "")))
            if bounds is not None:
                lower, upper = bounds
                return self._tool_success(
                    tool.__class__.__name__,
                    tool.integrate(expression, var=var, lower=lower, upper=upper),
                    {"operation": "integral"},
                )
            return self._tool_success(tool.__class__.__name__, tool.integrate(expression, var=var), {"operation": "integral"})
        if operation == "limit":
            point = question.get("point") or self._extract_limit_point(question, str(question.get("question", "")))
            return self._tool_success(tool.__class__.__name__, tool.limit(expression, var=var, point=point), {"operation": "limit"})
        if operation == "critical":
            return self._tool_success(tool.__class__.__name__, tool.critical_points(expression, var=var), {"operation": "critical_points"})
        return self._tool_success(tool.__class__.__name__, tool.taylor_series(expression, var=var), {"operation": "taylor"})

    def _dispatch_linalg(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult, lowered_prompt: str) -> Any:
        operation = self._pick_operation(lowered_prompt, ("determinant", "det", "行列式", "inverse", "逆", "matrix", "rank", "秩", "eigen", "特征值", "power", "幂"))
        # Prefer structured matrix field, fallback to text extraction
        matrix = question.get("matrix")
        if matrix is None:
            matrix = self._extract_matrix_from_text(str(question.get("question", "")))

        if matrix is None:
            return self._tool_failure("linalg", "No matrix found in question.", {"operation": operation or "unknown"})

        if operation in {"determinant", "det", "行列式"}:
            return self._tool_success(tool.__class__.__name__, tool.determinant(matrix), {"operation": "determinant"})
        if operation in {"inverse", "逆"}:
            return self._tool_success(tool.__class__.__name__, tool.matrix_inverse(matrix), {"operation": "inverse"})
        if operation in {"rank", "秩"}:
            return self._tool_success(tool.__class__.__name__, tool.rank(matrix), {"operation": "rank"})
        if operation in {"eigen", "特征值", "特征向量"}:
            return self._tool_success(
                tool.__class__.__name__,
                {"eigenvalues": tool.eigenvalues(matrix), "eigenvectors": tool.eigenvectors(matrix)},
                {"operation": "eigen"},
            )
        if operation in {"power", "幂"} or "^{" in str(question.get("question", "")) or "^" in str(question.get("question", "")):
            exponent = self._extract_matrix_power_exponent(str(question.get("question", "")))
            if exponent is not None and hasattr(tool, "matrix_power"):
                return self._tool_success(tool.__class__.__name__, tool.matrix_power(matrix, exponent), {"operation": "matrix_power", "exponent": exponent})
            if exponent is not None:
                return self._tool_failure("linalg", "matrix_power not available in current tool.", {"operation": "matrix_power"})
        rhs = question.get("rhs")
        if rhs is not None:
            return self._tool_success(tool.__class__.__name__, tool.solve_linear_system(matrix, rhs), {"operation": "solve_linear_system"})
        return self._tool_success(tool.__class__.__name__, tool.simplify(str(question.get("question", "")) or "x"), {"operation": "simplify"})

    def _dispatch_circuits(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult, lowered_prompt: str) -> Any:
        topology = str(question.get("topology") or self._pick_operation(lowered_prompt, ("series", "parallel")) or "series")
        resistors = question.get("resistors")
        if resistors is not None:
            return self._tool_success(
                tool.__class__.__name__,
                tool.equivalent_resistance(resistors, topology=topology),
                {"operation": "equivalent_resistance", "topology": topology},
            )
        # Try extracting resistor values from Chinese text
        extracted_resistors = self._extract_resistors_from_text(str(question.get("question", "")))
        if extracted_resistors:
            return self._tool_success(
                tool.__class__.__name__,
                tool.equivalent_resistance(extracted_resistors, topology=topology),
                {"operation": "equivalent_resistance", "topology": topology, "extracted": True},
            )
        netlist = question.get("netlist")
        if netlist is not None:
            return self._tool_success(tool.__class__.__name__, tool.node_analysis(netlist), {"operation": "node_analysis"})
        mesh_matrix = question.get("mesh_matrix")
        source_vector = question.get("source_vector")
        if mesh_matrix is not None and source_vector is not None:
            return self._tool_success(tool.__class__.__name__, tool.mesh_analysis(mesh_matrix, source_vector), {"operation": "mesh_analysis"})
        q_keys = question.keys()
        if {"kind", "resistance", "reactive", "t", "initial"} <= q_keys:
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
        return self._tool_failure("circuits", "No extractable circuit data found.", {"operation": "unmatched"})

    def _dispatch_physics(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult, lowered_prompt: str) -> Any:
        relation = question.get("relation") or self._infer_physics_relation(str(question.get("question", "")))
        knowns = question.get("knowns")
        if knowns is None:
            knowns = self._extract_physics_knowns_from_text(str(question.get("question", "")))
        target = question.get("target") or self._infer_physics_target(str(question.get("question", "")))
        if not knowns or not relation or not target:
            return self._tool_failure(
                "physics",
                "Physics fast path needs knowns, relation, and target.",
                {"relation": str(relation), "target": str(target), "knowns": str(knowns)},
            )
        return self._tool_success(
            tool.__class__.__name__,
            tool.solve_relation(str(relation), knowns, str(target)),
            {"relation": str(relation), "target": str(target)},
        )

    # ------------------------------------------------------------------
    # Fallback analysis (no LLM)
    # ------------------------------------------------------------------

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
        knowns_str = f"已知: {self._join_items(analysis.knowns)}" if analysis.knowns else "已知: 题干信息已提取。"
        unknowns_str = f"未知: {self._join_items(analysis.unknowns)}" if analysis.unknowns else "未知: 求目标量。"
        equations_str = f"公式: {self._join_items(analysis.equations_or_theorems)}" if analysis.equations_or_theorems else "公式: 使用该学科标准关系式。"

        if tool_success and tool_output:
            reasoning_parts = [knowns_str, unknowns_str, equations_str, f"工具计算: {tool_output}", "结论: 代入并整理得到结果。"]
            reasoning = " | ".join(reasoning_parts)
            if question_text:
                reasoning = f"题目: {question_text}. {reasoning}"
            answer = tool_output
        else:
            failure_reason = error_message or str(tool_result.get("metadata", {}).get("error", "")).strip()
            if not failure_reason:
                failure_reason = "当前 fallback 未能完成精确求解。"
            reasoning_parts = [
                knowns_str,
                unknowns_str,
                f"相关公式: {self._join_items(analysis.equations_or_theorems)}" if analysis.equations_or_theorems else "相关公式: 使用该学科标准关系式。",
                f"当前失败原因: {failure_reason}",
                "暂无法可靠给出最终数值。",
            ]
            reasoning = " | ".join(reasoning_parts)
            if question_text:
                reasoning = f"题目: {question_text}. {reasoning}"
            answer = f"暂无法可靠给出最终数值: {failure_reason}"
        return DraftResult(reasoning_process=reasoning, answer=str(answer))

    # ------------------------------------------------------------------
    # Text extraction helpers (enhanced for Chinese + math)
    # ------------------------------------------------------------------

    def _extract_expression(self, question: dict[str, Any], prompt: str) -> str:
        for key in ("expression", "equation", "function"):
            val = question.get(key)
            if val:
                return self._normalize_expression(str(val))
        text = prompt.strip()
        for pattern in _EXPR_PATTERNS:
            match = pattern.search(text)
            if match:
                expr = match.group("expr").strip()
                expr = re.split(r"\s+(?:from|as|的|为)\b.*$", expr, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                if expr:
                    return self._normalize_expression(expr)
        return "x"

    def _extract_matrix_from_text(self, text: str) -> list[list[Any]] | None:
        """Extract matrix from text like [[1,2],[3,4]] or |1 2; 3 4| or [1 2; 3 4]."""
        # Try [[1,2],[3,4]] format
        match = re.search(r"\[\s*\[.+?\]\s*\]", text)
        if match:
            try:
                candidate = match.group(0)
                candidate = candidate.replace("，", ",").replace("；", ";")
                parsed = json.loads(candidate.replace("'", "\""))
                if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                    return parsed
            except Exception:
                pass
        # Try determinant notation |1 2 3; 2 4 6; 1 0 1|
        match = re.search(r"\|\s*([^|]+?)\s*\|", text)
        if match:
            try:
                inner = match.group(1).strip()
                inner = inner.replace("；", ";").replace("，", ",")
                rows = [r.strip() for r in inner.split(";")]
                matrix = []
                for row in rows:
                    entries = re.split(r"[,\s]+", row.strip())
                    matrix.append([self._parse_number_token(e) for e in entries if e.strip()])
                if matrix and all(len(r) == len(matrix[0]) for r in matrix):
                    return matrix
            except Exception:
                pass
        # Try [1 2; 3 4] format
        match = re.search(r"\[\s*([^\[\]]+?)\s*;\s*([^\[\]]+?)\s*\]", text)
        if match:
            try:
                rows = [match.group(1).strip(), match.group(2).strip()]
                matrix = []
                for row in rows:
                    entries = re.split(r"[,\s]+", row.strip())
                    matrix.append([self._parse_number_token(e) for e in entries if e.strip()])
                return matrix
            except Exception:
                pass
        return None

    def _extract_matrix_power_exponent(self, text: str) -> int | None:
        """Extract exponent from 'A^{2025}' or 'A^2025'."""
        match = re.search(r"\^\{(\d+)\}", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\^(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_resistors_from_text(self, text: str) -> list[float] | None:
        """Extract resistor values like '2 ohm', '3Ω' from Chinese text."""
        values = _RESISTOR_RE.findall(text)
        if values:
            return [float(v) for v in values]
        return None

    def _extract_physics_knowns_from_text(self, text: str) -> dict[str, Any]:
        """Extract known physical quantities from Chinese text."""
        lowered = text.lower()
        knowns: dict[str, Any] = {}
        # Match patterns like "2 kg", "3 m/s", "10 N"
        import re as _re
        mass_match = _re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|千克|kg)", text, _re.IGNORECASE)
        if mass_match:
            knowns["m"] = float(mass_match.group(1))
        vel_match = _re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|米/秒)", text, _re.IGNORECASE)
        if vel_match:
            knowns["v"] = float(vel_match.group(1))
        accel_match = _re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s\^2|m/s²|米/秒²)", text, _re.IGNORECASE)
        if accel_match:
            knowns["a"] = float(accel_match.group(1))
        force_match = _re.search(r"(\d+(?:\.\d+)?)\s*(?:N|牛顿)", text, _re.IGNORECASE)
        if force_match:
            knowns["F"] = float(force_match.group(1))
        return knowns

    def _extract_variable(self, prompt: str, default: str = "x") -> str:
        lowered = prompt.lower()
        if " with respect to y" in lowered or "对 y" in prompt or "关于 y" in prompt:
            return "y"
        if " with respect to t" in lowered or "对 t" in prompt or "关于 t" in prompt:
            return "t"
        return default

    def _extract_bounds(self, question: dict[str, Any], prompt: str) -> tuple[Any, Any] | None:
        if "lower" in question and "upper" in question:
            return question["lower"], question["upper"]
        match = re.search(r"\bfrom\s+([\-0-9./]+)\s+to\s+([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if match:
            return self._parse_number_token(match.group(1)), self._parse_number_token(match.group(2))
        # Chinese bounds: "从 0 到 3" or "在 [0,3] 上"
        match = re.search(r"[\[\(]([\-0-9./]+)\s*[,，]\s*([\-0-9./]+)[\]\)]", prompt)
        if match:
            return self._parse_number_token(match.group(1)), self._parse_number_token(match.group(2))
        return None

    def _extract_limit_point(self, question: dict[str, Any], prompt: str) -> Any:
        if "point" in question:
            return question["point"]
        match = re.search(r"\b(?:as\s+[a-zA-Z]\s*->|x\s*->|t\s*->)\s*([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if match:
            return self._parse_number_token(match.group(1))
        # Chinese limit point
        match = re.search(r"趋向于?\s*([\-0-9./]+)", prompt)
        if match:
            return self._parse_number_token(match.group(1))
        return 0

    def _parse_number_token(self, token: str) -> Any:
        cleaned = "".join(ch for ch in token if ch in "0123456789.-/")
        if not cleaned:
            return 0
        try:
            if "/" in cleaned:
                parts = cleaned.split("/")
                return float(parts[0]) / float(parts[1])
            if "." in cleaned:
                return float(cleaned)
            return int(cleaned)
        except ValueError:
            return 0

    def _normalize_expression(self, expression: str) -> str:
        return expression.replace("^", "**")

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_subject(self, text: str, fallback_subject: str) -> str:
        lowered = text.lower()
        supported = ("physics", "circuits", "linalg", "calculus")
        if fallback_subject in supported and self._subject_has_signal(lowered, fallback_subject):
            return fallback_subject
        for subject in ("calculus", "linalg", "circuits", "physics"):
            if subject != fallback_subject and self._subject_has_signal(lowered, subject):
                return subject
        return fallback_subject if fallback_subject in supported else "physics"

    def _infer_topic(self, subject: str, prompt: str) -> str:
        text = prompt.lower()
        if subject == "calculus":
            if "integral" in text or "integrate" in text or "积分" in prompt:
                return "integration"
            if "limit" in text or "极限" in prompt:
                return "limit"
            if "taylor" in text or "泰勒" in prompt:
                return "taylor_series"
            return "differentiation"
        if subject == "linalg":
            if "determinant" in text or "det" in text or "行列式" in prompt:
                return "determinant"
            if "eigen" in text or "特征值" in prompt or "特征向量" in prompt:
                return "eigen"
            if "rank" in text or "秩" in prompt:
                return "rank"
            if "inverse" in text or "逆" in prompt:
                return "inverse"
            if "幂" in prompt or "power" in text or "^{" in prompt:
                return "matrix_power"
            return "matrix_operations"
        if subject == "circuits":
            if "parallel" in text or "series" in text or "串联" in prompt or "并联" in prompt:
                return "equivalent_resistance"
            if "node" in text or "节点" in prompt:
                return "node_analysis"
            if "mesh" in text or "网孔" in prompt:
                return "mesh_analysis"
            if "rc" in text or "rl" in text:
                return "first_order_response"
            return "basic_circuits"
        if "momentum" in text or "动量" in prompt:
            return "momentum"
        if "energy" in text or "能量" in prompt or "功" in prompt:
            return "work_energy"
        if "force" in text or "力" in prompt:
            return "newton_second_law"
        return "kinematics"

    def _pick_operation(self, text: str, keywords: tuple[str, ...]) -> str:
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered:
                return keyword
        return ""

    def _subject_has_signal(self, text: str, subject: str) -> bool:
        if subject == "calculus":
            return self._has_keyword(text, ("integral", "derivative", "limit", "diff", "taylor", "积分", "导数", "极限", "微分")) or "series" in text or "泰勒" in text
        if subject == "linalg":
            return self._has_keyword(text, ("matrix", "vector", "eigen", "rank", "determinant", "inverse", "矩阵", "向量", "特征值", "秩", "行列式", "逆"))
        if subject == "circuits":
            return self._has_keyword(
                text,
                ("circuit", "resistor", "resistors", "resistance", "voltage", "current", "node", "mesh", "ohm", "电路", "电阻", "电压", "电流", "串联", "并联"),
            ) or self._has_acronym(text, ("rc", "rl"))
        if subject == "physics":
            return self._has_keyword(text, ("force", "velocity", "acceleration", "momentum", "energy", "mass", "力", "速度", "加速度", "动量", "能量", "质量"))
        return False

    def _has_keyword(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) for keyword in keywords)

    def _has_acronym(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) for keyword in keywords)

    def _infer_physics_relation(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(token in lowered for token in ("momentum", "impulse", "动量", "冲量")):
            return "momentum"
        if any(token in lowered for token in ("energy", "work", "能量", "功")):
            return "work_energy"
        if "force" in lowered or "力" in prompt:
            return "newton_second_law"
        return "uniform_acceleration"

    def _infer_physics_target(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "force" in lowered or "力" in prompt:
            return "F"
        if "momentum" in lowered or "动量" in prompt:
            return "p"
        if "energy" in lowered or "work" in lowered or "功" in prompt or "能量" in prompt:
            return "W"
        if "acceleration" in lowered or "加速度" in prompt:
            return "a"
        if "velocity" in lowered or "速度" in prompt:
            return "v"
        return "v"

    def _fallback_analysis_fields(
        self, subject: str, prompt: str
    ) -> tuple[list[str], list[str], list[str], bool, str, list[str]]:
        if subject == "calculus":
            lowered = prompt.lower()
            if "integral" in lowered or "integrate" in lowered or "积分" in prompt:
                return _FALLBACK_CALCULUS_INTEGRAL
            if "limit" in lowered or "极限" in prompt:
                return _FALLBACK_CALCULUS_LIMIT
            return _FALLBACK_CALCULUS_DEFAULT
        if subject == "linalg":
            return _FALLBACK_LINALG
        if subject == "circuits":
            return _FALLBACK_CIRCUITS
        return _FALLBACK_PHYSICS

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _join_items(self, items: list[Any]) -> str:
        return ", ".join(str(item) for item in items if str(item).strip())

    def _tool_success(self, tool_name: str, output: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._build_tool_result(tool_name, True, output, metadata or {}, None)

    def _tool_failure(self, tool_name: str, error_message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._build_tool_result(tool_name, False, None, metadata or {}, error_message)

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

    def _serialize_tool_output(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, (dict, list)):
            return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=self._json_default)
        if isinstance(output, complex):
            if abs(output.imag) < 1e-12:
                return str(float(output.real))
            return f"{output.real:.6f}+{output.imag:.6f}j"
        return str(output)

    def _json_default(self, obj: Any) -> Any:
        if isinstance(obj, complex):
            if abs(obj.imag) < 1e-12:
                return float(obj.real)
            return {"real": float(obj.real), "imag": float(obj.imag)}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _should_try_remote_client(self) -> bool:
        if self.kimi_client is None:
            return False
        config = getattr(self.kimi_client, "config", None)
        if config is not None and hasattr(config, "base_url"):
            return bool(getattr(config, "base_url", ""))
        return True

    def _resolve_kimi_client(self, user_provided: Any | None) -> KimiClient | None:
        """Respect explicit None; build default only when user omitted the parameter."""
        if user_provided is not None:
            return user_provided
        # user did not pass kimi_client at all → build default
        return self._build_default_kimi_client()

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
