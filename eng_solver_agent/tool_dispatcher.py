"""Standalone tool dispatch layer.

Extracted from agent.py to break the tight coupling between
agent logic and tool execution.
"""

from __future__ import annotations

import json
import re
from typing import Any

from eng_solver_agent.debug_logger import log_tool_dispatch, log_tool_result, log_error
from eng_solver_agent.schemas import AnalyzeResult


# Pre-compiled regex patterns
_EXPR_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"(?:differentiate|derivative of|compute the derivative of|find the derivative of)\s+(?P<expr>.+?)(?:\s+as\b|\s+from\b|\s*$|[.?])",
        r"(?:integral of|compute the integral of|find the integral of)\s+(?P<expr>.+?)(?:\s+from\b|\s+as\b|\s*$|[.?])",
        r"(?:limit of|find the limit of|compute the limit of)\s+(?P<expr>.+?)(?:\s+as\b|\s*$|[.?])",
        r"(?:求|计算|求导|求微分|对)\s*(?P<expr>[^，。；]+?)(?:\s*的导数|\s*的积分|\s*的微分|(?:\s+as\b|\s+from\b|\s*$|[.。；]))",
        r"(?:求|计算)\s*(?P<expr>[^，。；]+?)(?:\s*的不定积分|\s*的定积分|(?:\s+as\b|\s+from\b|\s*$|[.。；]))",
        r"(?:求|计算)\s*(?P<expr>[^，。；]+?)(?:\s*的极限|(?:\s+as\b|\s*$|[.。；]))",
    ]
)


class ToolDispatcher:
    """Routes questions to the appropriate tool based on subject and content."""

    def __init__(self, tools: dict[str, Any]) -> None:
        self.tools = tools

    def dispatch(self, question: dict[str, Any], analysis: AnalyzeResult) -> dict[str, Any]:
        """Dispatch a question to the appropriate tool and return structured result."""
        tool = self.tools.get(analysis.subject)
        if tool is None:
            result = self._build_result(analysis.subject, False, None, {}, "No tool registered.")
            log_tool_result(analysis.subject, False, None, "No tool registered.")
            return result
        try:
            log_tool_dispatch(analysis.subject, "auto", {
                "question_id": question.get("question_id", "?"),
                "subject": analysis.subject,
                "topic": analysis.topic,
            })
            payload = self._do_dispatch(tool, question, analysis)
            if isinstance(payload, dict):
                result = self._build_result(
                    payload.get("tool_name", analysis.subject),
                    bool(payload.get("success", True)),
                    payload.get("output", ""),
                    payload.get("metadata", {}),
                    payload.get("error_message"),
                )
                log_tool_result(
                    payload.get("tool_name", analysis.subject),
                    bool(payload.get("success", True)),
                    payload.get("output", ""),
                    payload.get("error_message"),
                )
                return result
            result = self._build_result(analysis.subject, True, payload, {}, None)
            log_tool_result(analysis.subject, True, payload)
            return result
        except Exception as exc:
            log_error("ToolDispatcher", exc)
            return self._build_result(
                analysis.subject, False, None, {"error": type(exc).__name__}, f"{type(exc).__name__}: {exc}"
            )

    def _do_dispatch(self, tool: Any, question: dict[str, Any], analysis: AnalyzeResult) -> Any:
        subject = analysis.subject
        prompt = str(question.get("question", ""))
        lowered = prompt.lower()

        if subject == "calculus":
            return self._dispatch_calculus(tool, question, lowered)
        if subject == "linalg":
            return self._dispatch_linalg(tool, question, lowered)
        if subject == "circuits":
            return self._dispatch_circuits(tool, question, lowered)
        if subject == "physics":
            return self._dispatch_physics(tool, question, lowered)
        return self._tool_failure(subject, "No deterministic tool rule matched.")

    def _dispatch_calculus(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        # Detect series-related problems
        if any(k in lowered for k in ("级数", "series", "收敛域", "convergence", "一致收敛", "uniform convergence", "收敛半径", "radius of convergence")):
            return self._dispatch_series(tool, question, lowered)

        operation = self._pick_operation(lowered, ("derivative", "diff", "integral", "limit", "critical", "taylor"))
        expression = question.get("expression") or question.get("equation") or question.get("function")
        if expression is None:
            expression = self._extract_expression(question, str(question.get("question", "")))
        var = question.get("variable") or self._extract_variable(str(question.get("question", "")), default="x")

        try:
            if operation in {"derivative", "diff"}:
                return self._tool_success(tool.__class__.__name__, tool.diff(expression, var=var), {"operation": "diff"})
            if operation == "integral":
                bounds = self._extract_bounds(question, str(question.get("question", "")))
                if bounds is not None:
                    lower, upper = bounds
                    return self._tool_success(tool.__class__.__name__, tool.integrate(expression, var=var, lower=lower, upper=upper), {"operation": "integral"})
                return self._tool_success(tool.__class__.__name__, tool.integrate(expression, var=var), {"operation": "integral"})
            if operation == "limit":
                point = question.get("point") or self._extract_limit_point(question, str(question.get("question", "")))
                return self._tool_success(tool.__class__.__name__, tool.limit(expression, var=var, point=point), {"operation": "limit"})
            if operation == "critical":
                return self._tool_success(tool.__class__.__name__, tool.critical_points(expression, var=var), {"operation": "critical_points"})
            return self._tool_success(tool.__class__.__name__, tool.taylor_series(expression, var=var), {"operation": "taylor"})
        except Exception as exc:
            return self._tool_failure(tool.__class__.__name__, f"Calculus tool error: {type(exc).__name__}: {exc}", {"operation": operation or "unknown"})

    def _dispatch_series(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        """Handle series-related problems."""
        # Check if it's a proof problem (uniform convergence)
        if any(k in lowered for k in ("一致收敛", "uniform convergence", "证明", "prove", "考查")):
            return self._tool_failure(
                tool.__class__.__name__,
                "Series uniform convergence proofs require symbolic reasoning and are not supported by the calculus tool. Please use LLM mode.",
                {"operation": "series_uniform_convergence", "requires_llm": True},
            )

        # Try to extract the general term for convergence radius calculation
        q_text = str(question.get("question", ""))
        term = self._extract_series_term(q_text)

        if term and hasattr(tool, "series_convergence_radius"):
            try:
                result = tool.series_convergence_radius(term)
                return self._tool_success(tool.__class__.__name__, result, {"operation": "series_convergence_radius", "term": term})
            except Exception as exc:
                return self._tool_failure(tool.__class__.__name__, f"Failed to compute convergence radius: {exc}", {"operation": "series_convergence_radius", "term": term})

        return self._tool_failure(
            tool.__class__.__name__,
            "Could not extract series general term from question text. Series convergence domain calculation requires a parseable expression.",
            {"operation": "series_convergence_domain"},
        )

    def _extract_series_term(self, text: str) -> str | None:
        """Extract the general term of a series from question text."""
        # Try to match patterns like sum_{n=1}^{infty} term
        patterns = [
            r"sum_{n=\d+}\^{.*?}\s+(.*?)(?:\)\s+的收敛域|\)\s*$|\)\s+在区间)",
            r"级数.*?sum_{n=\d+}\^{.*?}\s+(.*?)(?:\)\s+的收敛域|\)\s*$|\)\s+在区间)",
            r"\\frac\{(.*?)\}\{(.*?)\}",  # Try to extract frac as a simple term
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                term = match.group(1).strip()
                # Clean up the term
                term = term.replace("^{n}", "**n").replace("^{x}", "**x")
                term = term.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\ln", "ln")
                term = term.replace("\\infty", "oo")
                return term
        return None

    def _dispatch_linalg(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        operation = self._pick_operation(lowered, ("determinant", "det", "行列式", "inverse", "逆", "matrix", "rank", "秩", "eigen", "特征值", "power", "幂"))
        matrix = question.get("matrix")
        if matrix is None:
            matrix = self._extract_matrix_from_text(str(question.get("question", "")))
        if matrix is None:
            return self._tool_failure("linalg", "No matrix found in question.")

        if operation in {"determinant", "det", "行列式"}:
            return self._tool_success(tool.__class__.__name__, tool.determinant(matrix), {"operation": "determinant"})
        if operation in {"inverse", "逆"}:
            return self._tool_success(tool.__class__.__name__, tool.matrix_inverse(matrix), {"operation": "inverse"})
        if operation in {"rank", "秩"}:
            return self._tool_success(tool.__class__.__name__, tool.rank(matrix), {"operation": "rank"})
        if operation in {"eigen", "特征值", "特征向量"}:
            return self._tool_success(tool.__class__.__name__, {"eigenvalues": tool.eigenvalues(matrix), "eigenvectors": tool.eigenvectors(matrix)}, {"operation": "eigen"})
        if operation in {"power", "幂"} or "^{" in str(question.get("question", "")) or "^" in str(question.get("question", "")):
            exponent = self._extract_matrix_power_exponent(str(question.get("question", "")))
            if exponent is not None and hasattr(tool, "matrix_power"):
                return self._tool_success(tool.__class__.__name__, tool.matrix_power(matrix, exponent), {"operation": "matrix_power", "exponent": exponent})
            if exponent is not None:
                return self._tool_failure("linalg", "matrix_power not available.", {"operation": "matrix_power"})
        rhs = question.get("rhs")
        if rhs is not None:
            return self._tool_success(tool.__class__.__name__, tool.solve_linear_system(matrix, rhs), {"operation": "solve_linear_system"})
        return self._tool_success(tool.__class__.__name__, tool.simplify(str(question.get("question", "")) or "x"), {"operation": "simplify"})

    def _dispatch_circuits(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        topology = str(question.get("topology") or self._pick_operation(lowered, ("series", "parallel")) or "series")
        resistors = question.get("resistors")
        if resistors is not None:
            return self._tool_success(tool.__class__.__name__, tool.equivalent_resistance(resistors, topology=topology), {"operation": "equivalent_resistance", "topology": topology})
        extracted = self._extract_resistors_from_text(str(question.get("question", "")))
        if extracted:
            return self._tool_success(tool.__class__.__name__, tool.equivalent_resistance(extracted, topology=topology), {"operation": "equivalent_resistance", "topology": topology, "extracted": True})
        netlist = question.get("netlist")
        if netlist is not None:
            return self._tool_success(tool.__class__.__name__, tool.node_analysis(netlist), {"operation": "node_analysis"})
        mesh_matrix = question.get("mesh_matrix")
        source_vector = question.get("source_vector")
        if mesh_matrix is not None and source_vector is not None:
            return self._tool_success(tool.__class__.__name__, tool.mesh_analysis(mesh_matrix, source_vector), {"operation": "mesh_analysis"})
        q_keys = question.keys()
        if {"kind", "resistance", "reactive", "t", "initial"} <= q_keys:
            return self._tool_success(tool.__class__.__name__, tool.first_order_response(question["kind"], resistance=question["resistance"], reactive=question["reactive"], t=question["t"], initial=question["initial"], final=question.get("final")), {"operation": "first_order_response"})
        return self._tool_failure("circuits", "No extractable circuit data found.")

    def _dispatch_physics(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        relation = question.get("relation") or self._infer_physics_relation(str(question.get("question", "")))
        knowns = question.get("knowns")
        if knowns is None:
            knowns = self._extract_physics_knowns_from_text(str(question.get("question", "")))
        target = question.get("target") or self._infer_physics_target(str(question.get("question", "")))
        if not knowns or not relation or not target:
            return self._tool_failure("physics", "Physics fast path needs knowns, relation, and target.", {"relation": str(relation), "target": str(target), "knowns": str(knowns)})
        return self._tool_success(tool.__class__.__name__, tool.solve_relation(str(relation), knowns, str(target)), {"relation": str(relation), "target": str(target)})

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_expression(self, question: dict[str, Any], prompt: str) -> str:
        for key in ("expression", "equation", "function"):
            val = question.get(key)
            if val:
                return str(val).replace("^", "**")
        text = prompt.strip()
        for pattern in _EXPR_PATTERNS:
            match = pattern.search(text)
            if match:
                expr = match.group("expr").strip()
                expr = re.split(r"\s+(?:from|as|的|为)\b.*$", expr, maxsplit=1, flags=re.IGNORECASE)[0].strip()
                if expr:
                    return expr.replace("^", "**")
        return "x"

    def _extract_matrix_from_text(self, text: str) -> list[list[Any]] | None:
        match = re.search(r"\[\s*\[.+?\]\s*\]", text)
        if match:
            try:
                candidate = match.group(0).replace("，", ",").replace("；", ";")
                parsed = json.loads(candidate.replace("'", "\""))
                if isinstance(parsed, list) and all(isinstance(row, list) for row in parsed):
                    return parsed
            except Exception:
                pass
        match = re.search(r"\|\s*([^|]+?)\s*\|", text)
        if match:
            try:
                inner = match.group(1).strip().replace("；", ";").replace("，", ",")
                rows = [r.strip() for r in inner.split(";")]
                matrix = []
                for row in rows:
                    entries = re.split(r"[,\s]+", row.strip())
                    matrix.append([self._parse_number(e) for e in entries if e.strip()])
                if matrix and all(len(r) == len(matrix[0]) for r in matrix):
                    return matrix
            except Exception:
                pass
        match = re.search(r"\[\s*([^\[\]]+?)\s*;\s*([^\[\]]+?)\s*\]", text)
        if match:
            try:
                rows = [match.group(1).strip(), match.group(2).strip()]
                matrix = []
                for row in rows:
                    entries = re.split(r"[,\s]+", row.strip())
                    matrix.append([self._parse_number(e) for e in entries if e.strip()])
                return matrix
            except Exception:
                pass
        return None

    def _extract_matrix_power_exponent(self, text: str) -> int | None:
        match = re.search(r"\^\{(\d+)\}", text)
        if match:
            return int(match.group(1))
        match = re.search(r"\^(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_resistors_from_text(self, text: str) -> list[float] | None:
        values = re.findall(r"(\d+)\s*[Ω欧姆]", text, flags=re.IGNORECASE)
        if values:
            return [float(v) for v in values]
        return None

    def _extract_physics_knowns_from_text(self, text: str) -> dict[str, Any]:
        knowns: dict[str, Any] = {}
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|千克)", text, re.IGNORECASE)
        if m:
            knowns["m"] = float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|米/秒)", text, re.IGNORECASE)
        if m:
            knowns["v"] = float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s\^2|m/s²|米/秒²)", text, re.IGNORECASE)
        if m:
            knowns["a"] = float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:N|牛顿)", text, re.IGNORECASE)
        if m:
            knowns["F"] = float(m.group(1))
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
        m = re.search(r"\bfrom\s+([\-0-9./]+)\s+to\s+([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if m:
            return self._parse_number(m.group(1)), self._parse_number(m.group(2))
        m = re.search(r"[\[\(]([\-0-9./]+)\s*[,，]\s*([\-0-9./]+)[\]\)]", prompt)
        if m:
            return self._parse_number(m.group(1)), self._parse_number(m.group(2))
        return None

    def _extract_limit_point(self, question: dict[str, Any], prompt: str) -> Any:
        if "point" in question:
            return question["point"]
        m = re.search(r"\b(?:as\s+[a-zA-Z]\s*->|x\s*->|t\s*->)\s*([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if m:
            return self._parse_number(m.group(1))
        m = re.search(r"趋向于?\s*([\-0-9./]+)", prompt)
        if m:
            return self._parse_number(m.group(1))
        return 0

    def _parse_number(self, token: str) -> Any:
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

    def _pick_operation(self, text: str, keywords: tuple[str, ...]) -> str:
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered:
                return keyword
        return ""

    def _infer_physics_relation(self, prompt: str) -> str:
        lowered = prompt.lower()
        if any(t in lowered for t in ("momentum", "impulse", "动量", "冲量")):
            return "momentum"
        if any(t in lowered for t in ("energy", "work", "能量", "功")):
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

    def _tool_success(self, tool_name: str, output: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"tool_name": tool_name, "success": True, "output": self._serialize(output), "metadata": metadata or {}, "error_message": None}

    def _tool_failure(self, tool_name: str, error_message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"tool_name": tool_name, "success": False, "output": "", "metadata": metadata or {}, "error_message": error_message}

    def _build_result(self, tool_name: str, success: bool, output: Any, metadata: dict[str, Any], error_message: str | None) -> dict[str, Any]:
        return {"tool_name": tool_name, "success": bool(success), "output": self._serialize(output), "metadata": metadata or {}, "error_message": error_message}

    def _serialize(self, output: Any) -> str:
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
