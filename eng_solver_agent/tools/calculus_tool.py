"""Calculus tool layer with a sympy-first interface and a safe fallback."""

from __future__ import annotations

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
            return str(self._sympy.simplify(expression))
        return poly_to_string(poly_from_ast(expression))

    def diff(self, expression: str, var: str = "x", order: int = 1, at: Any | None = None) -> str | float:
        self._validate_order(order)
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            result = self._sympy.diff(self._sympy.sympify(expression), symbol, order)
            if at is not None:
                return float(self._sympy.N(result.subs(symbol, at)))
            return str(result)

        poly = poly_from_ast(expression, var=var)
        derived = poly_diff(poly, order=order)
        if at is not None:
            return float(poly_eval(derived, at))
        return poly_to_string(derived, var=var)

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
            return str(result)

        poly = poly_from_ast(expression, var=var)
        if lower is not None or upper is not None:
            if lower is None or upper is None:
                raise ValueError("both lower and upper bounds are required for definite integrals")
            antiderivative = poly_integral(poly)
            upper_value = poly_eval(antiderivative, upper)
            lower_value = poly_eval(antiderivative, lower)
            return float(upper_value - lower_value)
        return poly_to_string(poly_integral(poly), var=var)

    def limit(self, expression: str, var: str = "x", point: Any = 0, direction: str = "both") -> float:
        self._validate_direction(direction)
        if self._sympy is not None:
            symbol = self._sympy.Symbol(var)
            dir_map = {"both": "+-", "left": "-", "right": "+"}
            return float(
                self._sympy.N(
                    self._sympy.limit(
                        self._sympy.sympify(expression),
                        symbol,
                        point,
                        dir=dir_map[direction],
                    )
                )
            )

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
            return {"center": center, "order": order, "series": str(series)}

        poly = poly_from_ast(expression, var=var)
        return taylor_series(poly, center=center, order=order, var=var)

    def _validate_order(self, order: int) -> None:
        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a non-negative integer")

    def _validate_direction(self, direction: str) -> None:
        if direction not in {"both", "left", "right"}:
            raise ValueError("direction must be 'both', 'left', or 'right'")
