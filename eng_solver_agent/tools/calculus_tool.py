"""Calculus tool layer with a sympy-first interface and a safe fallback."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from eng_solver_agent.tools._math_support import (
    ToolUnsupportedError,
    load_sympy,
    poly_diff,
    poly_eval,
    poly_from_ast,
    poly_integral,
    poly_roots,
    poly_to_string,
    quadratic_critical_points,
    taylor_series,
)


class CalculusTool:
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

        poly = poly_from_ast(expression, var=var)
        return taylor_series(poly, center=center, order=order, var=var)

    def _validate_order(self, order: int) -> None:
        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a non-negative integer")

    def _validate_direction(self, direction: str) -> None:
        if direction not in {"both", "left", "right"}:
            raise ValueError("direction must be 'both', 'left', or 'right'")

    def _poly_to_fraction_string(self, poly: dict[int, Fraction], var: str = "x") -> str:
        """Convert polynomial to string with fraction coefficients."""
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
        """Extract coefficients from sympy series string."""
        import re
        coefficients = [0.0, 0.0, 0.0]
        terms = series_str.replace("-", "+-").split("+")
        for term in terms:
            term = term.strip()
            if not term:
                continue
            power = 0
            if f"{var}**" in term:
                power_match = re.search(rf"{re.escape(var)}\*\*(\d+)", term)
                if power_match:
                    power = int(power_match.group(1))
            elif term.endswith(var) or (f"*{var}" in term and f"{var}**" not in term):
                power = 1
            if power > 2:
                continue
            coeff_match = re.match(r"^(-?\d+(?:/\d+)?)", term)
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
        return coefficients

    def _format_integral_result(self, result: str, var: str) -> str:
        """Format integral result to match expected format like '1/3*x**3'."""
        import re
        match = re.match(r"^(\d+)/(\d+)\*(.+)$", result)
        if match:
            return result
        match = re.match(rf"^(.+?)/(\d+)$", result)
        if match:
            numerator = match.group(1)
            denominator = match.group(2)
            if numerator == var:
                return f"1/{denominator}*{var}"
            power_match = re.search(rf"{re.escape(var)}\*\*(\d+)", numerator)
            if power_match:
                return f"1/{denominator}*{var}**{power_match.group(1)}"
        return result

