"""Unified numerical computation tool.

Integrates calculus, linear algebra, circuit analysis, and physics computation
into a single cohesive tool class while preserving all original functionality.
This follows the demo template pattern of a unified Math_Solver tool.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from eng_solver_agent.exceptions import ToolUnsupportedError
from eng_solver_agent.tools._math_support import (
    determinant,
    eigenpairs_2x2,
    inverse,
    load_sympy,
    poly_diff,
    poly_eval,
    poly_from_ast,
    poly_integral,
    poly_roots,
    poly_to_string,
    quadratic_critical_points,
    rank,
    solve_linear_system,
    taylor_series,
)


class NumericalComputationTool:
    """Unified numerical computation engine.

    Aggregates all deterministic calculation capabilities from the legacy
    CalculusTool, AlgebraTool, CircuitTool, and PhysicsTool into a single
    interface. The original tools are kept as internal delegates so that
    every method behaves exactly as before.
    """

    def __init__(self) -> None:
        self._calculus = _CalculusEngine()
        self._algebra = _AlgebraEngine()
        self._circuit = _CircuitEngine()
        self._physics = _PhysicsEngine()

    # ------------------------------------------------------------------
    # Unified entry points (demo-template style)
    # ------------------------------------------------------------------

    def solve(self, query: str) -> str:
        """Parse a natural-language or code-like query and attempt to solve it.

        Supports two styles:
        1. Python-style expression: "diff('x**2', 'x')"
        2. Natural-language hint: "derivative of x**2 with respect to x"
        """
        try:
            return self.compute_from_query(query)
        except Exception as exc:
            return f"[numerical-tool] 解析或计算失败: {type(exc).__name__}: {exc}"

    def compute(self, operation: str, **kwargs: Any) -> Any:
        """Unified compute dispatcher.

        Args:
            operation: The name of the operation to perform.
            **kwargs: Named arguments passed to the underlying method.

        Returns:
            The raw result from the delegated tool.
        """
        operation = operation.strip().lower()
        method = getattr(self, operation, None)
        if method is None or operation.startswith("_"):
            raise ValueError(f"不支持的操作: '{operation}'。可用操作见工具描述。")
        return method(**kwargs)

    def compute_from_query(self, query: str) -> str:
        """Parse a query string and execute the corresponding operation.

        Expected formats:
        - "diff(expression='x**2', var='x')"
        - "integrate(expression='x**2', var='x', lower=0, upper=1)"
        - "determinant(matrix=[[1,2],[3,4]])"
        - "equivalent_resistance(resistors=[10,20], topology='parallel')"
        """
        query = query.strip()
        if not query:
            raise ValueError("查询不能为空")

        # Try to extract operation name and keyword arguments
        match = re.match(r"(\w+)\s*\((.*)\)\s*$", query, re.DOTALL)
        if match:
            op_name = match.group(1)
            args_str = match.group(2).strip()
            kwargs = self._parse_kwargs(args_str)
            result = self.compute(op_name, **kwargs)
            return str(result)

        # Fallback: try to infer from natural language
        return self._infer_and_compute(query)

    # ------------------------------------------------------------------
    # Calculus delegation
    # ------------------------------------------------------------------

    def diff(self, expression: str, var: str = "x", order: int = 1, at: Any | None = None) -> str | float:
        return self._calculus.diff(expression, var=var, order=order, at=at)

    def integrate(
        self,
        expression: str,
        var: str = "x",
        lower: Any | None = None,
        upper: Any | None = None,
    ) -> str | float:
        return self._calculus.integrate(expression, var=var, lower=lower, upper=upper)

    def limit(self, expression: str, var: str = "x", point: Any = 0, direction: str = "both") -> float:
        return self._calculus.limit(expression, var=var, point=point, direction=direction)

    def critical_points(self, expression: str, var: str = "x") -> list[float]:
        return self._calculus.critical_points(expression, var=var)

    def taylor_series(self, expression: str, var: str = "x", center: Any = 0, order: int = 5) -> dict[str, Any]:
        return self._calculus.taylor_series(expression, var=var, center=center, order=order)

    def series_convergence_radius(self, expression: str, var: str = "x") -> dict[str, Any]:
        return self._calculus.series_convergence_radius(expression, var=var)

    def simplify(self, expression: str) -> str:
        return self._calculus.simplify(expression)

    # ------------------------------------------------------------------
    # Linear algebra delegation
    # ------------------------------------------------------------------

    def determinant(self, matrix: list[list[Any]]) -> float:
        return self._algebra.determinant(matrix)

    def matrix_inverse(self, matrix: list[list[Any]]) -> list[list[float]]:
        return self._algebra.matrix_inverse(matrix)

    def rank(self, matrix: list[list[Any]]) -> int:
        return self._algebra.rank(matrix)

    def eigenvalues(self, matrix: list[list[Any]]) -> list[Any]:
        return self._algebra.eigenvalues(matrix)

    def eigenvectors(self, matrix: list[list[Any]]) -> list[list[Any]]:
        return self._algebra.eigenvectors(matrix)

    def matrix_power(self, matrix: list[list[Any]], exponent: int) -> list[list[float]]:
        return self._algebra.matrix_power(matrix, exponent)

    def solve_linear_system(self, matrix: list[list[Any]], rhs: list[Any]) -> list[float]:
        return self._algebra.solve_linear_system(matrix, rhs)

    def linear_independence(self, vectors: list[list[Any]]) -> dict[str, Any]:
        return self._algebra.linear_independence(vectors)

    # ------------------------------------------------------------------
    # Circuit analysis delegation
    # ------------------------------------------------------------------

    def equivalent_resistance(self, resistors: list[float], topology: str = "series") -> float:
        return self._circuit.equivalent_resistance(resistors, topology=topology)

    def node_analysis(self, netlist: dict[str, Any], ground: str = "0") -> dict[str, float]:
        return self._circuit.node_analysis(netlist, ground=ground)

    def mesh_analysis(self, resistance_matrix: list[list[Any]], source_vector: list[Any]) -> list[float]:
        return self._circuit.mesh_analysis(resistance_matrix, source_vector)

    def first_order_response(
        self,
        kind: str,
        resistance: float,
        reactive: float,
        t: float,
        initial: float,
        final: float | None = None,
    ) -> dict[str, float]:
        return self._circuit.first_order_response(
            kind, resistance=resistance, reactive=reactive, t=t, initial=initial, final=final
        )

    # ------------------------------------------------------------------
    # Physics delegation
    # ------------------------------------------------------------------

    def solve_relation(self, relation: str, knowns: dict[str, Any], target: str) -> float:
        return self._physics.solve_relation(relation, knowns, target)

    def uniform_acceleration(self, knowns: dict[str, Any], target: str) -> float:
        return self._physics.solve_relation("uniform_acceleration", knowns, target)

    def newton_second_law(self, knowns: dict[str, Any], target: str) -> float:
        return self._physics.solve_relation("newton_second_law", knowns, target)

    def work_energy(self, knowns: dict[str, Any], target: str) -> float:
        return self._physics.solve_relation("work_energy", knowns, target)

    def momentum(self, knowns: dict[str, Any], target: str) -> float:
        return self._physics.solve_relation("momentum", knowns, target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_kwargs(self, args_str: str) -> dict[str, Any]:
        """Parse a string like "expression='x**2', var='x'" into a dict."""
        kwargs: dict[str, Any] = {}
        if not args_str:
            return kwargs
        # Use a safe eval-like approach: split by commas at top level
        parts = self._split_top_level(args_str)
        for part in parts:
            part = part.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            kwargs[key] = self._coerce_value(value)
        return kwargs

    def _split_top_level(self, text: str) -> list[str]:
        """Split a comma-separated string, respecting nested brackets."""
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for ch in text:
            if ch in "([{":
                depth += 1
                current.append(ch)
            elif ch in ")]}]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def _coerce_value(self, value: str) -> Any:
        """Coerce a string value to an appropriate Python type."""
        value = value.strip()
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none":
            return None
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        return value

    def _infer_and_compute(self, query: str) -> str:
        """Infer operation from natural-language query and execute."""
        lowered = query.lower()

        # Calculus inference
        if any(k in lowered for k in ("导数", "求导", "derivative", "differentiate", "diff")):
            expr = self._extract_expression(lowered)
            var = "y" if "对 y" in query or "with respect to y" in lowered else "x"
            return str(self.diff(expr, var=var))

        if any(k in lowered for k in ("积分", "integral", "integrate")):
            expr = self._extract_expression(lowered)
            var = "y" if "对 y" in query or "with respect to y" in lowered else "x"
            bounds = self._extract_bounds_from_text(lowered)
            if bounds:
                return str(self.integrate(expr, var=var, lower=bounds[0], upper=bounds[1]))
            return str(self.integrate(expr, var=var))

        if any(k in lowered for k in ("极限", "limit")):
            expr = self._extract_expression(lowered)
            point = self._extract_limit_point_from_text(lowered)
            return str(self.limit(expr, point=point))

        # Linear algebra inference
        if any(k in lowered for k in ("行列式", "determinant", "det")):
            matrix = self._extract_matrix_from_text(query)
            if matrix:
                return str(self.determinant(matrix))
            raise ValueError("未能从查询中提取矩阵")

        if any(k in lowered for k in ("逆矩阵", "inverse", "逆")):
            matrix = self._extract_matrix_from_text(query)
            if matrix:
                return str(self.matrix_inverse(matrix))
            raise ValueError("未能从查询中提取矩阵")

        if any(k in lowered for k in ("秩", "rank")):
            matrix = self._extract_matrix_from_text(query)
            if matrix:
                return str(self.rank(matrix))
            raise ValueError("未能从查询中提取矩阵")

        if any(k in lowered for k in ("特征值", "eigenvalue", "特征向量", "eigenvector")):
            matrix = self._extract_matrix_from_text(query)
            if matrix:
                return json.dumps(
                    {"eigenvalues": self.eigenvalues(matrix), "eigenvectors": self.eigenvectors(matrix)},
                    ensure_ascii=False,
                )
            raise ValueError("未能从查询中提取矩阵")

        # Circuit inference
        if any(k in lowered for k in ("等效电阻", "equivalent resistance", "电阻", "resistor")):
            resistors = self._extract_resistors_from_text(query)
            topology = "parallel" if any(k in lowered for k in ("并联", "parallel")) else "series"
            if resistors:
                return str(self.equivalent_resistance(resistors, topology=topology))
            raise ValueError("未能从查询中提取电阻值")

        # Physics inference
        if any(k in lowered for k in ("牛顿", "newton", "力", "force")):
            knowns = self._extract_physics_knowns(query)
            return str(self.newton_second_law(knowns, "F"))

        if any(k in lowered for k in ("动能", "功", "work", "energy")):
            knowns = self._extract_physics_knowns(query)
            return str(self.work_energy(knowns, "W"))

        if any(k in lowered for k in ("动量", "momentum")):
            knowns = self._extract_physics_knowns(query)
            return str(self.momentum(knowns, "p"))

        raise ValueError(f"无法从查询中推断操作: {query[:50]}...")

    def _extract_expression(self, text: str) -> str:
        match = re.search(r"([\dxa-zA-Z\+\-\*/\^\(\)\.]+)\s*(?:的导数|的积分|的极限|derivative|integral|limit)", text)
        if match:
            return match.group(1).replace("^", "**")
        match = re.search(r"(?:求|计算| differentiate|find the derivative of|integrate)\s+([\dxa-zA-Z\+\-\*/\^\(\)\.]+)", text)
        if match:
            return match.group(1).replace("^", "**")
        return "x"

    def _extract_bounds_from_text(self, text: str) -> tuple[float, float] | None:
        match = re.search(r"from\s+([\-\d\.]+)\s+to\s+([\-\d\.]+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
        match = re.search(r"[\[\(]([\-\d\.]+)\s*,\s*([\-\d\.]+)[\]\)]", text)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None

    def _extract_limit_point_from_text(self, text: str) -> float:
        match = re.search(r"(?:趋向|趋近|as|->|→)\s*([\-\d\.]+|∞|infinity)", text, re.IGNORECASE)
        if match:
            token = match.group(1)
            if token in ("∞", "infinity"):
                return float("inf")
            return float(token)
        return 0

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
                    matrix.append([float(e) if "." in e else int(e) for e in entries if e.strip()])
                if matrix and all(len(r) == len(matrix[0]) for r in matrix):
                    return matrix
            except Exception:
                pass
        return None

    def _extract_resistors_from_text(self, text: str) -> list[float] | None:
        values = re.findall(r"(\d+(?:\.\d+)?)\s*[Ω欧姆]", text, flags=re.IGNORECASE)
        if values:
            return [float(v) for v in values]
        return None

    def _extract_physics_knowns(self, text: str) -> dict[str, Any]:
        knowns: dict[str, Any] = {}
        patterns = [
            (r"(\d+(?:\.\d+)?)\s*(?:kg|千克)", "m"),
            (r"(\d+(?:\.\d+)?)\s*(?:m/s|米/秒)", "v"),
            (r"(\d+(?:\.\d+)?)\s*(?:m/s\^2|m/s²|米/秒²)", "a"),
            (r"(\d+(?:\.\d+)?)\s*(?:N|牛顿)", "F"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                knowns[key] = float(match.group(1))
        return knowns

    # ------------------------------------------------------------------
    # Exec-based symbolic/numeric engine (example.py pattern)
    # ------------------------------------------------------------------

    def execute_code(self, code: str) -> str:
        """Execute Python/SymPy code in a sandboxed environment.

        This follows the example.py pattern, exposing sympy, math, and numpy
        via a restricted globals dict with stdout capture.

        Args:
            code: A Python code string. The caller should use print() to
                output results so they are captured and returned.

        Returns:
            Captured stdout from the executed code, or an error message.
        """
        return _ExecEngine().execute(code)


# ------------------------------------------------------------------------------
# Legacy engine classes (inlined here so the old standalone files can be removed)
# ------------------------------------------------------------------------------

class _CalculusEngine:
    """Internal calculus engine — was CalculusTool."""

    def __init__(self) -> None:
        self._sympy = load_sympy()

    def solve(self, expression: str) -> str:
        try:
            return self.simplify(expression)
        except ToolUnsupportedError as exc:
            return f"[sympy-unavailable] {exc}"

    def simplify(self, expression: str) -> str:
        if self._sympy is not None:
            result = str(self._sympy.simplify(expression))
            if result == expression:
                raise ToolUnsupportedError(f"unsupported symbolic expression: {expression}")
            return result
        return poly_to_string(poly_from_ast(expression))

    def diff(self, expression: str, var: str = "x", order: int = 1, at: Any | None = None) -> str | float:
        self._validate_order(order)
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            expr = self._sympy.sympify(expression)
            result = self._sympy.diff(expr, symbol, order)
            if at is not None:
                return float(self._sympy.N(result.subs(symbol, at)))
            result_str = str(result)
            if result_str == expression:
                raise ToolUnsupportedError(f"unsupported symbolic expression: {expression}")
            return result_str

        poly = poly_from_ast(expression, var=var)
        derived = poly_diff(poly, order=order)
        if at is not None:
            return float(poly_eval(derived, at))
        result = poly_to_string(derived, var=var)
        if result == expression:
            raise ToolUnsupportedError(f"unsupported symbolic expression: {expression}")
        return result

    def integrate(
        self,
        expression: str,
        var: str = "x",
        lower: Any | None = None,
        upper: Any | None = None,
    ) -> str | float:
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            result = self._sympy.integrate(self._sympy.sympify(expression), symbol)
            if lower is not None and upper is not None:
                definite = self._sympy.integrate(self._sympy.sympify(expression), (symbol, lower, upper))
                return float(self._sympy.N(definite))
            return self._format_integral_result(str(result), var)

        poly = poly_from_ast(expression, var=var)
        if lower is not None or upper is not None:
            if lower is None or upper is None:
                raise ValueError("both lower and upper bounds are required for definite integrals")
            antiderivative = poly_integral(poly)
            upper_value = poly_eval(antiderivative, upper)
            lower_value = poly_eval(antiderivative, lower)
            return float(upper_value - lower_value)
        return self._poly_to_fraction_string(poly_integral(poly), var=var)

    def limit(self, expression: str, var: str = "x", point: Any = 0, direction: str = "both") -> float:
        self._validate_direction(direction)
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            dir_param = "+-" if direction == "both" else direction
            return float(self._sympy.N(self._sympy.limit(self._sympy.sympify(expression), symbol, point, dir=dir_param)))

        poly = poly_from_ast(expression, var=var)
        return float(poly_eval(poly, point))

    def critical_points(self, expression: str, var: str = "x") -> list[float]:
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            derivative = self._sympy.diff(self._sympy.sympify(expression), symbol)
            roots = self._sympy.solve(derivative, symbol)
            return [float(self._sympy.N(root)) for root in roots]

        poly = poly_from_ast(expression, var=var)
        return quadratic_critical_points(poly)

    def taylor_series(self, expression: str, var: str = "x", center: Any = 0, order: int = 5) -> dict[str, Any]:
        self._validate_order(order)
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            series = self._sympy.series(self._sympy.sympify(expression), symbol, center, order + 1).removeO()
            series_str = str(series)
            coefficients = self._extract_coefficients_from_series(series_str, var)
            return {"center": center, "order": order, "series": series_str, "coefficients": coefficients}

        try:
            poly = poly_from_ast(expression, var=var)
            return taylor_series(poly, center=center, order=order, var=var)
        except Exception:
            raise ToolUnsupportedError(f"taylor_series requires sympy for non-polynomial expressions: {expression}")

    def series_convergence_radius(self, expression: str, var: str = "x") -> dict[str, Any]:
        if self._sympy is None:
            raise ToolUnsupportedError("series_convergence_radius requires sympy")

        n = self._sympy.Symbol("n")
        x = self._sympy.Symbol(var)
        expr = self._sympy.sympify(expression)

        a_n = expr
        a_n1 = self._sympy.sympify(expression.replace("n", "(n+1)"))
        ratio = self._sympy.limit(self._sympy.Abs(a_n / a_n1), n, self._sympy.oo)

        return {
            "method": "ratio",
            "convergence_radius": float(ratio) if ratio.is_number else str(ratio),
            "expression": expression,
        }

    def _validate_order(self, order: int) -> None:
        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a non-negative integer")

    def _validate_direction(self, direction: str) -> None:
        if direction not in {"both", "left", "right"}:
            raise ValueError("direction must be 'both', 'left', or 'right'")

    def _poly_to_fraction_string(self, poly: dict[int, Fraction], var: str = "x") -> str:
        if not poly:
            return "0"
        pieces: list[str] = []
        for power in sorted(poly, reverse=True):
            coeff = poly[power]
            sign = "-" if coeff < 0 else "+"
            magnitude = -coeff if coeff < 0 else coeff
            if power == 0:
                body = f"{magnitude.numerator}/{magnitude.denominator}" if magnitude.denominator != 1 else f"{magnitude.numerator}"
            elif power == 1:
                if magnitude.denominator == 1:
                    body = f"{magnitude.numerator}*{var}" if magnitude.numerator != 1 else var
                else:
                    body = f"{magnitude.numerator}/{magnitude.denominator}*{var}"
            else:
                if magnitude.denominator == 1:
                    body = f"{magnitude.numerator}*{var}**{power}" if magnitude.numerator != 1 else f"{var}**{power}"
                else:
                    body = f"{magnitude.numerator}/{magnitude.denominator}*{var}**{power}"
            pieces.append((sign, body))

        first_sign, first_body = pieces[0]
        out = f"-{first_body}" if first_sign == "-" else first_body
        for sign, body in pieces[1:]:
            out += f" {sign} {body}"
        return out

    def _extract_coefficients_from_series(self, series_str: str, var: str) -> list[float]:
        import re as _re
        coefficients: dict[int, float] = {}
        terms = series_str.replace("-", "+-").split("+")
        for term in terms:
            term = term.strip()
            if not term:
                continue
            power = 0
            if f"{var}**" in term:
                power_match = _re.search(rf"{_re.escape(var)}\*\*(\d+)", term)
                if power_match:
                    power = int(power_match.group(1))
            elif term.endswith(var) or (f"*{var}" in term and f"{var}**" not in term):
                power = 1
            coeff_match = _re.match(r"^(-?\d+(?:/\d+)?)", term)
            if coeff_match:
                coeff_str = coeff_match.group(1)
                if "/" in coeff_str:
                    num, den = coeff_str.split("/")
                    coeff = float(int(num)) / float(int(den))
                else:
                    coeff = float(coeff_str)
            elif term == var or term.endswith(f"*{var}") or term.startswith(f"{var}**"):
                coeff = 1.0
            elif term.startswith("-") and (var in term):
                coeff = -1.0
            else:
                coeff = 1.0 if term and term[0] != "-" else -1.0
            coefficients[power] = coeff
        if not coefficients:
            return []
        max_power = max(coefficients)
        return [coefficients.get(p, 0.0) for p in range(max_power + 1)]

    def _format_integral_result(self, result: str, var: str) -> str:
        import re as _re
        match = _re.match(r"^(\d+)/(\d+)\*(.+)$", result)
        if match:
            return result
        match = _re.match(rf"^(.+?)/(\d+)$", result)
        if match:
            numerator = match.group(1)
            denominator = match.group(2)
            if numerator == var:
                return f"1/{denominator}*{var}"
            power_match = _re.search(rf"{_re.escape(var)}\*\*(\d+)", numerator)
            if power_match:
                return f"1/{denominator}*{var}**{power_match.group(1)}"
        return result


class _AlgebraEngine:
    """Internal linear-algebra engine — was AlgebraTool."""

    def __init__(self) -> None:
        self._sympy = load_sympy()

    def solve(self, expression: str) -> str:
        try:
            return self.simplify(expression)
        except ToolUnsupportedError as exc:
            return f"[sympy-unavailable] {exc}"

    def solve_linear_system(self, matrix: list[list[Any]], rhs: list[Any]) -> list[float]:
        if self._sympy is not None:
            symbols = self._sympy.symbols(f"x0:{len(rhs)}")
            equations = []
            for row, value in zip(matrix, rhs):
                equations.append(
                    self._sympy.Eq(
                        sum(self._sympy.sympify(entry) * symbol for entry, symbol in zip(row, symbols)),
                        self._sympy.sympify(value),
                    )
                )
            solution = self._sympy.solve(equations, symbols, dict=True)
            if not solution:
                raise ToolUnsupportedError("system has no unique solution")
            return [float(solution[0][symbol]) for symbol in symbols]
        return solve_linear_system(matrix, rhs)

    def matrix_inverse(self, matrix: list[list[Any]]) -> list[list[float]]:
        if self._sympy is not None:
            mat = self._sympy.Matrix(matrix)
            return [[float(value) for value in row] for row in mat.inv().tolist()]
        return inverse(matrix)

    def determinant(self, matrix: list[list[Any]]) -> float:
        if self._sympy is not None:
            return float(self._sympy.Matrix(matrix).det())
        return determinant(matrix)

    def rank(self, matrix: list[list[Any]]) -> int:
        if self._sympy is not None:
            return int(self._sympy.Matrix(matrix).rank())
        return rank(matrix)

    def eigenvalues(self, matrix: list[list[Any]]) -> list[Any]:
        if self._sympy is not None:
            values = self._sympy.Matrix(matrix).eigenvals()
            out: list[Any] = []
            for value, multiplicity in values.items():
                out.extend([self._sympy.N(value)] * int(multiplicity))
            return [self._coerce_scalar(value) for value in out]
        eigenvalues, _ = eigenpairs_2x2(matrix)
        return eigenvalues

    def eigenvectors(self, matrix: list[list[Any]]) -> list[list[Any]]:
        if self._sympy is not None:
            vectors = self._sympy.Matrix(matrix).eigenvects()
            out: list[list[Any]] = []
            for _, _, basis in vectors:
                for vector in basis:
                    out.append([self._coerce_scalar(value) for value in vector])
            return out
        _, eigenvectors = eigenpairs_2x2(matrix)
        return eigenvectors

    def simplify(self, expression: str) -> str:
        if self._sympy is not None:
            result = str(self._sympy.simplify(expression))
            if result == expression:
                raise ToolUnsupportedError(f"unsupported symbolic expression: {expression}")
            return result
        poly = poly_from_ast(expression)
        result = poly_to_string(poly)
        if result == expression:
            raise ToolUnsupportedError(f"unsupported symbolic expression: {expression}")
        return result

    def matrix_power(self, matrix: list[list[Any]], exponent: int) -> list[list[float]]:
        if self._sympy is not None:
            mat = self._sympy.Matrix(matrix)
            result = mat ** exponent
            return [[float(value) for value in row] for row in result.tolist()]
        size = len(matrix)
        if size == 0 or any(len(row) != size for row in matrix):
            raise ToolUnsupportedError("matrix_power requires a square matrix")
        result = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        base = [[float(entry) for entry in row] for row in matrix]
        power = exponent
        while power > 0:
            if power & 1:
                result = self._matrix_mult(result, base)
            base = self._matrix_mult(base, base)
            power >>= 1
        return result

    def _matrix_mult(self, a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        n = len(a)
        m = len(b[0])
        p = len(b)
        result = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                total = 0.0
                for k in range(p):
                    total += a[i][k] * b[k][j]
                result[i][j] = total
        return result

    def linear_independence(self, vectors: list[list[Any]]) -> dict[str, Any]:
        if not vectors or not vectors[0]:
            return {"independent": False, "reason": "empty vector set"}
        if self._sympy is not None:
            mat = self._sympy.Matrix(vectors)
            r = mat.rank()
            num_vectors = len(vectors)
            independent = r == num_vectors
            return {
                "independent": independent,
                "rank": int(r),
                "num_vectors": num_vectors,
                "relation": None if independent else "vectors are linearly dependent",
            }
        r = rank(vectors)
        num_vectors = len(vectors)
        independent = r == num_vectors
        return {
            "independent": independent,
            "rank": r,
            "num_vectors": num_vectors,
            "relation": None if independent else "vectors are linearly dependent",
        }

    def _coerce_scalar(self, value: Any) -> Any:
        try:
            numeric = complex(value)
        except Exception:
            return str(value)
        if abs(numeric.imag) < 1e-12:
            return float(numeric.real)
        return numeric


class _CircuitEngine:
    """Internal circuit-analysis engine — was CircuitTool."""

    def __init__(self) -> None:
        self._sympy = load_sympy()

    def solve(self, expression: str) -> str:
        return f"[circuit-tool] {expression}"

    def equivalent_resistance(self, resistors: list[float], topology: str = "series") -> float:
        if not resistors:
            raise ValueError("resistors must not be empty")
        topology = topology.lower()
        if topology == "series":
            return float(sum(resistors))
        if topology == "parallel":
            if any(r == 0 for r in resistors):
                return 0.0
            return float(1.0 / sum(1.0 / float(r) for r in resistors))
        raise ValueError("topology must be 'series' or 'parallel'")

    def node_analysis(self, netlist: dict[str, Any], ground: str = "0") -> dict[str, float]:
        nodes = [str(node) for node in netlist.get("nodes", []) if str(node) != ground]
        components = netlist.get("components", [])
        if not nodes:
            return {ground: 0.0}

        index = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        rhs = [0.0 for _ in range(size)]

        voltage_sources: list[dict[str, Any]] = []
        for component in components:
            ctype = str(component.get("type", "")).lower()
            if ctype == "resistor":
                n1 = str(component["n1"])
                n2 = str(component["n2"])
                value = float(component["value"])
                if value == 0:
                    raise ValueError("resistor value cannot be zero")
                conductance = 1.0 / value
                self._stamp_conductance(matrix, index, n1, n2, conductance)
            elif ctype == "current_source":
                n_plus = str(component.get("n_plus", component.get("n1")))
                n_minus = str(component.get("n_minus", component.get("n2")))
                value = float(component["value"])
                self._stamp_current(rhs, index, n_plus, n_minus, value)
            elif ctype == "voltage_source":
                voltage_sources.append(component)
            else:
                raise ToolUnsupportedError(f"unsupported netlist component: {ctype}")

        self._apply_voltage_sources(voltage_sources, matrix, rhs, index, ground)

        voltages = solve_linear_system(matrix, rhs)
        result = {ground: 0.0}
        for node, idx in index.items():
            result[node] = voltages[idx]
        return result

    def mesh_analysis(self, resistance_matrix: list[list[Any]], source_vector: list[Any]) -> list[float]:
        if self._sympy is not None:
            matrix = self._sympy.Matrix(resistance_matrix)
            rhs = self._sympy.Matrix(source_vector)
            solution = matrix.LUsolve(rhs)
            return [float(self._sympy.N(value)) for value in solution]
        return solve_linear_system(resistance_matrix, source_vector)

    def first_order_response(
        self,
        kind: str,
        resistance: float,
        reactive: float,
        t: float,
        initial: float,
        final: float | None = None,
    ) -> dict[str, float]:
        kind_normalized = kind.strip().upper()
        if kind_normalized == "RC":
            tau = float(resistance) * float(reactive)
        elif kind_normalized == "RL":
            if resistance == 0:
                raise ValueError("resistance cannot be zero for an RL response")
            tau = float(reactive) / float(resistance)
        else:
            raise ValueError("kind must be 'RC' or 'RL'")

        final_value = float(initial if final is None else final)
        value = final_value + (float(initial) - final_value) * math.exp(-float(t) / tau)
        return {"tau": tau, "value": value}

    def _stamp_conductance(
        self,
        matrix: list[list[float]],
        index: dict[str, int],
        n1: str,
        n2: str,
        conductance: float,
    ) -> None:
        i = index.get(n1)
        j = index.get(n2)
        if i is not None:
            matrix[i][i] += conductance
        if j is not None:
            matrix[j][j] += conductance
        if i is not None and j is not None:
            matrix[i][j] -= conductance
            matrix[j][i] -= conductance

    def _stamp_current(
        self,
        rhs: list[float],
        index: dict[str, int],
        n_plus: str,
        n_minus: str,
        value: float,
    ) -> None:
        plus = index.get(n_plus)
        minus = index.get(n_minus)
        if plus is not None:
            rhs[plus] -= value
        if minus is not None:
            rhs[minus] += value

    def _apply_voltage_sources(
        self,
        sources: list[dict[str, Any]],
        matrix: list[list[float]],
        rhs: list[float],
        index: dict[str, int],
        ground: str,
    ) -> None:
        for source in sources:
            n_plus = str(source.get("n_plus", source.get("n1")))
            n_minus = str(source.get("n_minus", source.get("n2")))
            value = float(source["value"])
            plus_idx = index.get(n_plus)
            minus_idx = index.get(n_minus)

            if n_minus == ground and plus_idx is not None:
                matrix[plus_idx] = [0.0] * len(matrix)
                matrix[plus_idx][plus_idx] = 1.0
                rhs[plus_idx] = value
            elif n_plus == ground and minus_idx is not None:
                matrix[minus_idx] = [0.0] * len(matrix)
                matrix[minus_idx][minus_idx] = 1.0
                rhs[minus_idx] = -value
            elif plus_idx is not None and minus_idx is not None:
                approx_resistance = 1e-9
                conductance = 1.0 / approx_resistance
                self._stamp_conductance(matrix, index, n_plus, n_minus, conductance)
                self._stamp_current(rhs, index, n_plus, n_minus, value / approx_resistance)
            else:
                raise ToolUnsupportedError("voltage source with floating node is not supported")


class _PhysicsEngine:
    """Internal physics engine — was PhysicsTool."""

    def __init__(self) -> None:
        self._sympy = load_sympy()

    def solve(self, expression: str) -> str:
        return f"[physics-tool] {expression}"

    def solve_relation(self, relation: str, knowns: dict[str, Any], target: str) -> float:
        normalized = relation.strip().lower()
        target_raw = target.strip()
        target_lower = target_raw.lower()
        if normalized in {"uniform_acceleration", "kinematics", "constant_acceleration"}:
            return self._solve_uniform_acceleration(knowns, target_raw, target_lower)
        if normalized in {"newton_second_law", "newton2", "f_ma"}:
            return self._solve_newton_second_law(knowns, target_raw, target_lower)
        if normalized in {"work_energy", "energy"}:
            return self._solve_work_energy(knowns, target_raw, target_lower)
        if normalized in {"momentum", "impulse"}:
            return self._solve_momentum(knowns, target_raw, target_lower)
        raise ToolUnsupportedError(f"unsupported physics relation: {relation}")

    def _solve_uniform_acceleration(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        v0 = knowns.get("v0")
        v = knowns.get("v")
        a = knowns.get("a")
        t = knowns.get("t")
        s = knowns.get("s")

        if target in {"v", "final_velocity"}:
            if v0 is None or a is None or t is None:
                raise ValueError("uniform_acceleration target 'v' requires v0, a, and t")
            return float(v0 + a * t)
        if target in {"s", "x", "displacement"}:
            if v0 is not None and a is not None and t is not None:
                return float(v0 * t + 0.5 * a * t * t)
            if v is not None and v0 is not None and a is not None:
                return float((v * v - v0 * v0) / (2 * a))
            raise ValueError("uniform_acceleration target 's' requires either (v0, a, t) or (v, v0, a)")
        if target in {"a", "acceleration"}:
            if v is None or v0 is None or t is None:
                raise ValueError("uniform_acceleration target 'a' requires v, v0, and t")
            return float((v - v0) / t)
        if target in {"t", "time"}:
            if v is not None and v0 is not None and a is not None:
                if a == 0:
                    raise ValueError("acceleration cannot be zero when solving for time")
                return float((v - v0) / a)
            if s is not None and v0 is not None and a is not None:
                return float(self._solve_quadratic_time(a, v0, s))
            raise ValueError("uniform_acceleration target 't' requires either (v, v0, a) or (s, v0, a)")
        raise ToolUnsupportedError(f"unsupported kinematics target: {target_raw}")

    def _solve_quadratic_time(self, a: Any, v0: Any, s: Any) -> float:
        if a == 0:
            if v0 == 0:
                raise ValueError("insufficient information to solve for time")
            return float(s / v0)
        discriminant = v0 * v0 + 2 * a * s
        if discriminant < 0:
            raise ValueError("no real solution for time")
        root = math.sqrt(float(discriminant))
        t1 = (-float(v0) + root) / float(a)
        t2 = (-float(v0) - root) / float(a)
        candidates = [value for value in (t1, t2) if value >= 0]
        if not candidates:
            raise ValueError("no non-negative time solution")
        return min(candidates)

    def _solve_newton_second_law(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        F = knowns.get("F")
        m = knowns.get("m")
        a = knowns.get("a")

        if target in {"f"}:
            if m is None or a is None:
                raise ValueError("Newton's second law target 'F' requires m and a")
            return float(m * a)
        if target == "m":
            if F is None or a is None:
                raise ValueError("Newton's second law target 'm' requires F and a")
            if a == 0:
                raise ValueError("acceleration cannot be zero when solving for mass")
            return float(F / a)
        if target == "a":
            if F is None or m is None:
                raise ValueError("Newton's second law target 'a' requires F and m")
            if m == 0:
                raise ValueError("mass cannot be zero when solving for acceleration")
            return float(F / m)
        raise ToolUnsupportedError(f"unsupported Newton target: {target_raw}")

    def _solve_work_energy(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        m = knowns.get("m")
        v0 = knowns.get("v0")
        v = knowns.get("v")
        F = knowns.get("F")
        s = knowns.get("s")
        theta = knowns.get("theta", 0)
        theta_rad = math.radians(float(theta))

        if target in {"w"}:
            if m is not None and v0 is not None and v is not None:
                return float(0.5 * m * (v * v - v0 * v0))
            if F is not None and s is not None:
                return float(F * s * math.cos(theta_rad))
            raise ValueError("work-energy target 'W' requires either (m, v0, v) or (F, s, theta)")
        if target in {"f"}:
            if s is None:
                raise ValueError("work-energy target 'F' requires s")
            work = knowns.get("W")
            if work is None:
                raise ValueError("work-energy target 'F' requires W")
            if s == 0:
                raise ValueError("distance cannot be zero when solving for force")
            return float(work / (s * math.cos(theta_rad)))
        raise ToolUnsupportedError(f"unsupported work-energy target: {target_raw}")

    def _solve_momentum(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        m = knowns.get("m")
        v = knowns.get("v")
        p = knowns.get("p")
        F = knowns.get("F")
        dt = knowns.get("dt")

        if target == "p":
            if m is None or v is None:
                raise ValueError("momentum target 'p' requires m and v")
            return float(m * v)
        if target == "v":
            if p is None or m is None:
                raise ValueError("momentum target 'v' requires p and m")
            if m == 0:
                raise ValueError("mass cannot be zero when solving for velocity")
            return float(p / m)
        if target == "impulse":
            if F is None or dt is None:
                raise ValueError("momentum target 'impulse' requires F and dt")
            return float(F * dt)
        raise ToolUnsupportedError(f"unsupported momentum target: {target_raw}")


class _ExecEngine:
    """Sandboxed Python/SymPy execution engine — follows example.py pattern."""

    def execute(self, code: str) -> str:
        from io import StringIO

        safe_globals = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "complex": complex,
                "dict": dict,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "pow": pow,
                "print": print,
                "range": range,
                "round": round,
                "set": set,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            },
            "json": __import__("json"),
            "math": __import__("math"),
            "np": __import__("numpy"),
            "sympy": __import__("sympy"),
            "diff": __import__("sympy").diff,
            "integrate": __import__("sympy").integrate,
            "limit": __import__("sympy").limit,
            "oo": __import__("sympy").oo,
            "pi": __import__("sympy").pi,
            "simplify": __import__("sympy").simplify,
            "solve": __import__("sympy").solve,
            "sqrt": __import__("sympy").sqrt,
            "symbols": __import__("sympy").symbols,
        }

        import threading

        old_stdout = __import__("sys").stdout
        __import__("sys").stdout = redirected = StringIO()
        result_container = {"output": None, "error": None, "done": False}

        def _exec_target():
            try:
                exec(code, safe_globals)
                __import__("sys").stdout = old_stdout
                output = redirected.getvalue().strip()
                result_container["output"] = output if output else "代码执行成功，但没有输出。请在代码中使用 print() 输出结果。"
            except Exception as exc:
                __import__("sys").stdout = old_stdout
                result_container["error"] = f"代码执行出错: {type(exc).__name__}: {exc}"
            finally:
                result_container["done"] = True

        thread = threading.Thread(target=_exec_target, daemon=True)
        thread.start()
        thread.join(timeout=5)

        if not result_container["done"]:
            __import__("sys").stdout = old_stdout
            return "代码执行超时 (5秒限制)"

        if result_container["error"] is not None:
            return result_container["error"]
        return result_container["output"]
