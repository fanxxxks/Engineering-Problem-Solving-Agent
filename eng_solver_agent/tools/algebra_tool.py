"""Algebra and linear algebra tool layer.

The implementation prefers sympy when it is installed. On this machine sympy
is absent, so the module falls back to a smaller but explicit exact/numeric
subset and raises `ToolUnsupportedError` for unsupported symbolic inputs.
"""

from __future__ import annotations

from typing import Any

from eng_solver_agent.tools._math_support import (
    ToolUnsupportedError,
    determinant,
    eigenpairs_2x2,
    inverse,
    load_sympy,
    poly_from_ast,
    poly_to_string,
    rank,
    solve_linear_system,
)


class AlgebraTool:
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
            return str(self._sympy.simplify(expression))
        poly = poly_from_ast(expression)
        return poly_to_string(poly)

    def _coerce_scalar(self, value: Any) -> Any:
        try:
            numeric = complex(value)
        except Exception:
            return str(value)
        if abs(numeric.imag) < 1e-12:
            return float(numeric.real)
        return numeric

