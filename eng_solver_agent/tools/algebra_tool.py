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
        """Compute matrix^n using repeated squaring."""
        if self._sympy is not None:
            mat = self._sympy.Matrix(matrix)
            result = mat ** exponent
            return [[float(value) for value in row] for row in result.tolist()]
        # Fallback using numpy-like manual multiplication
        size = len(matrix)
        if size == 0 or any(len(row) != size for row in matrix):
            raise ToolUnsupportedError("matrix_power requires a square matrix")
        # Initialize result as identity
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
        """Check if a set of vectors is linearly independent."""
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
        from eng_solver_agent.tools._math_support import rank as _rank
        r = _rank(vectors)
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

