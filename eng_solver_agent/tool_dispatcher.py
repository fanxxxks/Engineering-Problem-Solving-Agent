"""Standalone tool dispatch layer — builds sympy code strings and delegates to NumericalComputationTool.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from eng_solver_agent.debug_logger import log_tool_dispatch, log_tool_result, log_error
from eng_solver_agent.schemas import AnalyzeResult


# Pre-compiled regex patterns for expression extraction
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
    """Routes questions to the compute tool, building sympy code strings for execution."""

    def __init__(self, tools: dict[str, Any]) -> None:
        self.tools = tools

    def dispatch(self, question: dict[str, Any], analysis: AnalyzeResult) -> dict[str, Any]:
        """Dispatch a question to the compute tool and return structured result."""
        tool = self.tools.get("compute")
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

    # ------------------------------------------------------------------
    # Route to subject-specific dispatcher
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Calculus dispatch — builds sympy code
    # ------------------------------------------------------------------

    def _dispatch_calculus(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        if any(k in lowered for k in ("级数", "series", "收敛域", "convergence", "一致收敛", "uniform convergence", "收敛半径", "radius of convergence")):
            return self._dispatch_series(tool, question, lowered)

        operation = self._pick_operation(lowered, ("derivative", "diff", "integral", "limit", "critical", "taylor"))
        expression = question.get("expression") or question.get("equation") or question.get("function")
        if expression is None:
            expression = self._extract_expression(question, str(question.get("question", "")))
        var = question.get("variable") or self._extract_variable(str(question.get("question", "")), default="x")

        try:
            if operation in {"derivative", "diff"}:
                code = (
                    f"x = sympy.Symbol('{var}')\n"
                    f"expr = sympy.sympify('{self._escape(expression)}')\n"
                    f"result = sympy.diff(expr, x)\n"
                    f"print(result)"
                )
                return self._run_tool(tool, code, "diff")

            if operation == "integral":
                bounds = self._extract_bounds(question, str(question.get("question", "")))
                if bounds is not None:
                    lower, upper = bounds
                    code = (
                        f"x = sympy.Symbol('{var}')\n"
                        f"expr = sympy.sympify('{self._escape(expression)}')\n"
                        f"result = sympy.integrate(expr, (x, {lower}, {upper}))\n"
                        f"print(sympy.N(result))"
                    )
                else:
                    code = (
                        f"x = sympy.Symbol('{var}')\n"
                        f"expr = sympy.sympify('{self._escape(expression)}')\n"
                        f"result = sympy.integrate(expr, x)\n"
                        f"print(result)"
                    )
                return self._run_tool(tool, code, "integral")

            if operation == "limit":
                point = question.get("point") or self._extract_limit_point(question, str(question.get("question", "")))
                code = (
                    f"x = sympy.Symbol('{var}')\n"
                    f"expr = sympy.sympify('{self._escape(expression)}')\n"
                    f"result = sympy.limit(expr, x, {point})\n"
                    f"print(result)"
                )
                return self._run_tool(tool, code, "limit")

            if operation == "critical":
                code = (
                    f"x = sympy.Symbol('{var}')\n"
                    f"expr = sympy.sympify('{self._escape(expression)}')\n"
                    f"deriv = sympy.diff(expr, x)\n"
                    f"roots = sympy.solve(deriv, x)\n"
                    f"print([float(sympy.N(r)) for r in roots])"
                )
                return self._run_tool(tool, code, "critical_points")

            # taylor
            code = (
                f"x = sympy.Symbol('{var}')\n"
                f"expr = sympy.sympify('{self._escape(expression)}')\n"
                f"series = sympy.series(expr, x, 0, 6).removeO()\n"
                f"print(series)"
            )
            return self._run_tool(tool, code, "taylor")

        except Exception as exc:
            return self._tool_failure("calculus", f"Calculus tool error: {type(exc).__name__}: {exc}", {"operation": operation or "unknown"})

    # ------------------------------------------------------------------
    # Series dispatch
    # ------------------------------------------------------------------

    def _dispatch_series(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        if any(k in lowered for k in ("一致收敛", "uniform convergence", "证明", "prove", "考查")):
            return self._tool_failure(
                "calculus",
                "Series uniform convergence proofs require symbolic reasoning and are not supported by the calculus tool. Please use LLM mode.",
                {"operation": "series_uniform_convergence", "requires_llm": True},
            )

        q_text = str(question.get("question", ""))
        term = self._extract_series_term(q_text)

        if term:
            try:
                code = (
                    f"n = sympy.Symbol('n')\n"
                    f"a_n = sympy.sympify('{self._escape(term)}')\n"
                    f"limit_expr = sympy.limit(sympy.Abs(a_n / a_n.subs(n, n+1)), n, sympy.oo)\n"
                    f"R = limit_expr if limit_expr.is_number else str(limit_expr)\n"
                    f"print(R)"
                )
                return self._run_tool(tool, code, "series_convergence_radius")
            except Exception as exc:
                return self._tool_failure("calculus", f"Failed to compute convergence radius: {exc}", {"operation": "series_convergence_radius", "term": term})

        return self._tool_failure(
            "calculus",
            "Could not extract series general term from question text.",
            {"operation": "series_convergence_domain"},
        )

    # ------------------------------------------------------------------
    # Linear algebra dispatch — builds sympy code
    # ------------------------------------------------------------------

    def _dispatch_linalg(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        operation = self._pick_operation(lowered, ("determinant", "det", "行列式", "inverse", "逆", "matrix", "rank", "秩", "eigen", "特征值", "power", "幂"))
        matrix = question.get("matrix")
        if matrix is None:
            matrix = self._extract_matrix_from_text(str(question.get("question", "")))
        if matrix is None:
            return self._tool_failure("linalg", "No matrix found in question.")

        matrix_str = json.dumps(matrix)

        if operation in {"determinant", "det", "行列式"}:
            code = f"M = sympy.Matrix({matrix_str})\nprint(M.det())"
            return self._run_tool(tool, code, "determinant")

        if operation in {"inverse", "逆"}:
            code = f"M = sympy.Matrix({matrix_str})\nprint(M.inv())"
            return self._run_tool(tool, code, "inverse")

        if operation in {"rank", "秩"}:
            code = f"M = sympy.Matrix({matrix_str})\nprint(M.rank())"
            return self._run_tool(tool, code, "rank")

        if operation in {"eigen", "特征值", "特征向量"}:
            code = (
                f"M = sympy.Matrix({matrix_str})\n"
                f"evals = [sympy.N(v) for v in M.eigenvals()]\n"
                f"evecs = M.eigenvects()\n"
                f"print('eigenvalues:', evals)\n"
                f"print('eigenvectors:', evecs)"
            )
            return self._run_tool(tool, code, "eigen")

        if operation in {"power", "幂"} or "^{" in str(question.get("question", "")) or "^" in str(question.get("question", "")):
            exponent = self._extract_matrix_power_exponent(str(question.get("question", "")))
            if exponent is not None:
                code = f"M = sympy.Matrix({matrix_str})\nprint(M ** {exponent})"
                return self._run_tool(tool, code, "matrix_power")
            return self._tool_failure("linalg", "matrix_power requires an exponent.", {"operation": "matrix_power"})

        rhs = question.get("rhs")
        if rhs is not None:
            rhs_str = json.dumps(rhs)
            code = (
                f"M = sympy.Matrix({matrix_str})\n"
                f"b = sympy.Matrix({rhs_str})\n"
                f"print(M.LUsolve(b))"
            )
            return self._run_tool(tool, code, "solve_linear_system")

        # fallback: simplify
        code = f"print(sympy.simplify(sympy.Matrix({matrix_str})))"
        return self._run_tool(tool, code, "simplify")

    # ------------------------------------------------------------------
    # Circuit dispatch — simple arithmetic (no sympy needed)
    # ------------------------------------------------------------------

    def _dispatch_circuits(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        topology = str(question.get("topology") or self._pick_operation(lowered, ("series", "parallel")) or "series")
        resistors = question.get("resistors")
        if resistors is None:
            resistors = self._extract_resistors_from_text(str(question.get("question", "")))

        if resistors:
            code = self._build_eq_resistance_code(resistors, topology)
            return self._run_tool(tool, code, "equivalent_resistance")

        netlist = question.get("netlist")
        if netlist is not None:
            return self._tool_failure("circuits", "Node/mesh analysis requires LLM-generated sympy code.", {"operation": "node_analysis"})

        mesh_matrix = question.get("mesh_matrix")
        source_vector = question.get("source_vector")
        if mesh_matrix is not None and source_vector is not None:
            m_str = json.dumps(mesh_matrix)
            v_str = json.dumps(source_vector)
            code = f"M = sympy.Matrix({m_str})\nb = sympy.Matrix({v_str})\nprint(M.LUsolve(b))"
            return self._run_tool(tool, code, "mesh_analysis")

        q_keys = question.keys()
        if {"kind", "resistance", "reactive", "t", "initial"} <= q_keys:
            return self._dispatch_first_order_response(tool, question)

        return self._tool_failure("circuits", "No extractable circuit data found.")

    @staticmethod
    def _build_eq_resistance_code(resistors: list[float], topology: str) -> str:
        r_str = json.dumps(resistors)
        if topology == "parallel":
            return f"R = {r_str}\nprint(1 / sum(1/r for r in R))"
        return f"R = {r_str}\nprint(sum(R))"

    def _dispatch_first_order_response(self, tool: Any, question: dict[str, Any]) -> Any:
        kind = str(question["kind"]).strip().upper()
        resistance = float(question["resistance"])
        reactive = float(question["reactive"])
        t = float(question["t"])
        initial = float(question["initial"])
        final = float(question.get("final", initial))
        tau = resistance * reactive if kind == "RC" else reactive / resistance
        value = final + (initial - final) * math.exp(-t / tau)
        return self._tool_success("circuits", json.dumps({"tau": tau, "value": value}), {"operation": "first_order_response"})

    # ------------------------------------------------------------------
    # Physics dispatch — simple arithmetic (no sympy needed)
    # ------------------------------------------------------------------

    def _dispatch_physics(self, tool: Any, question: dict[str, Any], lowered: str) -> Any:
        relation = question.get("relation") or self._infer_physics_relation(str(question.get("question", "")))
        knowns = question.get("knowns")
        if knowns is None:
            knowns = self._extract_physics_knowns_from_text(str(question.get("question", "")))
        target = question.get("target") or self._infer_physics_target(str(question.get("question", "")))
        if not knowns or not relation or not target:
            return self._tool_failure("physics", "Physics fast path needs knowns, relation, and target.", {"relation": str(relation), "target": str(target), "knowns": str(knowns)})
        try:
            result = self._solve_physics_relation(str(relation), knowns, str(target))
            return self._tool_success("physics", result, {"relation": str(relation), "target": str(target)})
        except Exception as exc:
            return self._tool_failure("physics", f"Physics computation error: {exc}", {"relation": str(relation)})

    # ------------------------------------------------------------------
    # Physics relation solver (inlined from deleted _PhysicsEngine)
    # ------------------------------------------------------------------

    def _solve_physics_relation(self, relation: str, knowns: dict[str, Any], target: str) -> float:
        normalized = relation.strip().lower()
        target_lower = target.strip().lower()
        if normalized in {"uniform_acceleration", "kinematics", "constant_acceleration"}:
            return self._solve_uniform_acceleration(knowns, target, target_lower)
        if normalized in {"newton_second_law", "newton2", "f_ma"}:
            return self._solve_newton_second_law(knowns, target, target_lower)
        if normalized in {"work_energy", "energy"}:
            return self._solve_work_energy(knowns, target, target_lower)
        if normalized in {"momentum", "impulse"}:
            return self._solve_momentum(knowns, target, target_lower)
        raise ValueError(f"unsupported physics relation: {relation}")

    def _solve_uniform_acceleration(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        v0 = knowns.get("v0"); v = knowns.get("v"); a = knowns.get("a")
        t = knowns.get("t"); s = knowns.get("s")
        if target in {"v", "final_velocity"}:
            if v0 is None or a is None or t is None:
                raise ValueError("target 'v' requires v0, a, and t")
            return float(v0 + a * t)
        if target in {"s", "x", "displacement"}:
            if v0 is not None and a is not None and t is not None:
                return float(v0 * t + 0.5 * a * t * t)
            if v is not None and v0 is not None and a is not None:
                return float((v * v - v0 * v0) / (2 * a))
            raise ValueError("target 's' requires (v0,a,t) or (v,v0,a)")
        if target in {"a", "acceleration"}:
            if v is None or v0 is None or t is None:
                raise ValueError("target 'a' requires v, v0, and t")
            return float((v - v0) / t)
        if target in {"t", "time"}:
            if v is not None and v0 is not None and a is not None and a != 0:
                return float((v - v0) / a)
            if s is not None and v0 is not None and a is not None:
                disc = v0 * v0 + 2 * a * s
                if disc < 0: raise ValueError("no real solution for time")
                root = math.sqrt(float(disc))
                candidates = [val for val in ((-float(v0) + root) / float(a), (-float(v0) - root) / float(a)) if val >= 0]
                if candidates: return min(candidates)
                raise ValueError("no non-negative time solution")
            raise ValueError("target 't' requires (v,v0,a) or (s,v0,a)")
        raise ValueError(f"unsupported kinematics target: {target_raw}")

    def _solve_newton_second_law(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        F = knowns.get("F"); m = knowns.get("m"); a = knowns.get("a")
        if target == "f":
            if m is None or a is None: raise ValueError("target 'F' requires m and a")
            return float(m * a)
        if target == "m":
            if F is None or a is None or a == 0: raise ValueError("target 'm' requires F and nonzero a")
            return float(F / a)
        if target == "a":
            if F is None or m is None or m == 0: raise ValueError("target 'a' requires F and nonzero m")
            return float(F / m)
        raise ValueError(f"unsupported Newton target: {target_raw}")

    def _solve_work_energy(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        m = knowns.get("m"); v0 = knowns.get("v0"); v = knowns.get("v")
        F = knowns.get("F"); s = knowns.get("s"); theta = knowns.get("theta", 0)
        theta_rad = math.radians(float(theta))
        if target == "w":
            if m is not None and v0 is not None and v is not None:
                return float(0.5 * m * (v * v - v0 * v0))
            if F is not None and s is not None:
                return float(F * s * math.cos(theta_rad))
            raise ValueError("target 'W' requires (m,v0,v) or (F,s,theta)")
        if target == "f":
            if s is None: raise ValueError("target 'F' requires s")
            work = knowns.get("W")
            if work is None: raise ValueError("target 'F' requires W")
            if s == 0: raise ValueError("distance cannot be zero")
            return float(work / (s * math.cos(theta_rad)))
        raise ValueError(f"unsupported work-energy target: {target_raw}")

    def _solve_momentum(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        m = knowns.get("m"); v = knowns.get("v"); p = knowns.get("p")
        F = knowns.get("F"); dt = knowns.get("dt")
        if target == "p":
            if m is None or v is None: raise ValueError("target 'p' requires m and v")
            return float(m * v)
        if target == "v":
            if p is None or m is None or m == 0: raise ValueError("target 'v' requires p and nonzero m")
            return float(p / m)
        if target == "impulse":
            if F is None or dt is None: raise ValueError("target 'impulse' requires F and dt")
            return float(F * dt)
        raise ValueError(f"unsupported momentum target: {target_raw}")

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
                parsed = json.loads(candidate.replace("'", '"'))
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
        if m: knowns["m"] = float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s|米/秒)", text, re.IGNORECASE)
        if m: knowns["v"] = float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m/s\^2|m/s²|米/秒²)", text, re.IGNORECASE)
        if m: knowns["a"] = float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:N|牛顿)", text, re.IGNORECASE)
        if m: knowns["F"] = float(m.group(1))
        return knowns

    def _extract_variable(self, prompt: str, default: str = "x") -> str:
        lowered = prompt.lower()
        if " with respect to y" in lowered or "对 y" in prompt or "关于 y" in prompt: return "y"
        if " with respect to t" in lowered or "对 t" in prompt or "关于 t" in prompt: return "t"
        return default

    def _extract_bounds(self, question: dict[str, Any], prompt: str) -> tuple[Any, Any] | None:
        if "lower" in question and "upper" in question:
            return question["lower"], question["upper"]
        m = re.search(r"\bfrom\s+([\-0-9./]+)\s+to\s+([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if m: return self._parse_number(m.group(1)), self._parse_number(m.group(2))
        m = re.search(r"[\[\(]([\-0-9./]+)\s*[,，]\s*([\-0-9./]+)[\]\)]", prompt)
        if m: return self._parse_number(m.group(1)), self._parse_number(m.group(2))
        return None

    def _extract_limit_point(self, question: dict[str, Any], prompt: str) -> Any:
        if "point" in question: return question["point"]
        m = re.search(r"\b(?:as\s+[a-zA-Z]\s*->|x\s*->|t\s*->)\s*([\-0-9./]+)", prompt, flags=re.IGNORECASE)
        if m: return self._parse_number(m.group(1))
        m = re.search(r"趋向于?\s*([\-0-9./]+)", prompt)
        if m: return self._parse_number(m.group(1))
        return 0

    def _extract_series_term(self, text: str) -> str | None:
        patterns = [
            r"sum_{n=\d+}\^{.*?}\s+(.*?)(?:\)\s+的收敛域|\)\s*$|\)\s+在区间)",
            r"级数.*?sum_{n=\d+}\^{.*?}\s+(.*?)(?:\)\s+的收敛域|\)\s*$|\)\s+在区间)",
            r"\\frac\{(.*?)\}\{(.*?)\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                term = match.group(1).strip()
                term = term.replace("^{n}", "**n").replace("^{x}", "**x")
                term = term.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\ln", "ln")
                term = term.replace("\\infty", "oo")
                return term
        return None

    def _parse_number(self, token: str) -> Any:
        cleaned = "".join(ch for ch in token if ch in "0123456789.-/")
        if not cleaned: return 0
        try:
            if "/" in cleaned:
                parts = cleaned.split("/")
                return float(parts[0]) / float(parts[1])
            if "." in cleaned: return float(cleaned)
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
        if any(t in lowered for t in ("momentum", "impulse", "动量", "冲量")): return "momentum"
        if any(t in lowered for t in ("energy", "work", "能量", "功")): return "work_energy"
        if "force" in lowered or "力" in prompt: return "newton_second_law"
        return "uniform_acceleration"

    def _infer_physics_target(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "force" in lowered or "力" in prompt: return "F"
        if "momentum" in lowered or "动量" in prompt: return "p"
        if "energy" in lowered or "work" in lowered or "功" in prompt or "能量" in prompt: return "W"
        if "acceleration" in lowered or "加速度" in prompt: return "a"
        if "velocity" in lowered or "速度" in prompt: return "v"
        return "v"

    @staticmethod
    def _escape(s: str) -> str:
        """Escape backslashes and quotes for safe embedding in Python code strings."""
        return s.replace("\\", "\\\\").replace("'", "\\'")

    def _run_tool(self, tool: Any, code: str, operation: str) -> dict[str, Any]:
        """Execute code via the compute tool and wrap the result."""
        try:
            output = tool.compute(code)
            return self._tool_success("compute", output, {"operation": operation})
        except Exception as exc:
            return self._tool_failure("compute", f"{type(exc).__name__}: {exc}", {"operation": operation})

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _tool_success(self, tool_name: str, output: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"tool_name": tool_name, "success": True, "output": self._serialize(output), "metadata": metadata or {}, "error_message": None}

    def _tool_failure(self, tool_name: str, error_message: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"tool_name": tool_name, "success": False, "output": "", "metadata": metadata or {}, "error_message": error_message}

    def _build_result(self, tool_name: str, success: bool, output: Any, metadata: dict[str, Any], error_message: str | None) -> dict[str, Any]:
        return {"tool_name": tool_name, "success": bool(success), "output": self._serialize(output), "metadata": metadata or {}, "error_message": error_message}

    def _serialize(self, output: Any) -> str:
        if output is None: return ""
        if isinstance(output, (dict, list)):
            return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=self._json_default)
        if isinstance(output, complex):
            if abs(output.imag) < 1e-12: return str(float(output.real))
            return f"{output.real:.6f}+{output.imag:.6f}j"
        return str(output)

    def _json_default(self, obj: Any) -> Any:
        if isinstance(obj, complex):
            if abs(obj.imag) < 1e-12: return float(obj.real)
            return {"real": float(obj.real), "imag": float(obj.imag)}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
