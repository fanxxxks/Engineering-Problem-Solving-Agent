"""Shared math helpers for the tool layer.

The current environment does not have the real `sympy` package installed.
These helpers provide small, explicit fallbacks for the subset of operations
the project needs, and they raise clear errors for unsupported symbolic cases
instead of fabricating results.
"""

from __future__ import annotations

import ast
import cmath
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable


class ToolUnsupportedError(NotImplementedError):
    """Raised when an operation needs sympy or a broader solver."""


def load_sympy() -> Any | None:
    """Return sympy if installed, otherwise None."""

    try:
        import sympy as sp  # type: ignore

        return sp
    except Exception:
        return None


def _to_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid numeric inputs")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"Unsupported numeric type: {type(value)!r}")


def _cleanup_poly(poly: dict[int, Fraction]) -> dict[int, Fraction]:
    return {power: coeff for power, coeff in poly.items() if coeff != 0}


def poly_from_constant(value: Any) -> dict[int, Fraction]:
    return _cleanup_poly({0: _to_fraction(value)})


def poly_add(left: dict[int, Fraction], right: dict[int, Fraction], sign: int = 1) -> dict[int, Fraction]:
    out: dict[int, Fraction] = defaultdict(Fraction)
    for power, coeff in left.items():
        out[power] += coeff
    for power, coeff in right.items():
        out[power] += coeff * sign
    return _cleanup_poly(dict(out))


def poly_mul(left: dict[int, Fraction], right: dict[int, Fraction]) -> dict[int, Fraction]:
    out: dict[int, Fraction] = defaultdict(Fraction)
    for lp, lc in left.items():
        for rp, rc in right.items():
            out[lp + rp] += lc * rc
    return _cleanup_poly(dict(out))


def poly_pow(poly: dict[int, Fraction], exponent: int) -> dict[int, Fraction]:
    if exponent < 0:
        raise ToolUnsupportedError("negative polynomial powers are not supported without sympy")
    result = {0: Fraction(1)}
    base = dict(poly)
    power = exponent
    while power:
        if power & 1:
            result = poly_mul(result, base)
        base = poly_mul(base, base)
        power >>= 1
    return _cleanup_poly(result)


def poly_div_const(poly: dict[int, Fraction], divisor: Any) -> dict[int, Fraction]:
    scalar = _to_fraction(divisor)
    if scalar == 0:
        raise ZeroDivisionError("division by zero in polynomial expression")
    return _cleanup_poly({power: coeff / scalar for power, coeff in poly.items()})


def poly_diff(poly: dict[int, Fraction], order: int = 1) -> dict[int, Fraction]:
    result = dict(poly)
    for _ in range(order):
        derived: dict[int, Fraction] = {}
        for power, coeff in result.items():
            if power == 0:
                continue
            derived[power - 1] = derived.get(power - 1, Fraction(0)) + coeff * power
        result = _cleanup_poly(derived)
    return result


def poly_integral(poly: dict[int, Fraction]) -> dict[int, Fraction]:
    integrated: dict[int, Fraction] = {}
    for power, coeff in poly.items():
        integrated[power + 1] = coeff / Fraction(power + 1)
    return _cleanup_poly(integrated)


def poly_eval(poly: dict[int, Fraction], x_value: Any) -> Fraction | float:
    x = _to_fraction(x_value)
    total = Fraction(0)
    for power, coeff in poly.items():
        total += coeff * (x ** power)
    return total


def poly_to_string(poly: dict[int, Fraction], var: str = "x") -> str:
    if not poly:
        return "0"
    pieces: list[str] = []
    for power in sorted(poly, reverse=True):
        coeff = poly[power]
        sign = "-" if coeff < 0 else "+"
        magnitude = -coeff if coeff < 0 else coeff
        if power == 0:
            body = f"{magnitude}"
        elif power == 1:
            body = var if magnitude == 1 else f"{magnitude}*{var}"
        else:
            body = f"{var}**{power}" if magnitude == 1 else f"{magnitude}*{var}**{power}"
        pieces.append((sign, body))

    first_sign, first_body = pieces[0]
    out = f"-{first_body}" if first_sign == "-" else first_body
    for sign, body in pieces[1:]:
        out += f" {sign} {body}"
    return out


def poly_from_ast(expression: str, var: str = "x") -> dict[int, Fraction]:
    node = ast.parse(expression, mode="eval")
    return _poly_visit(node.body, var=var)


def _poly_visit(node: ast.AST, var: str) -> dict[int, Fraction]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return poly_from_constant(node.value)
        raise ToolUnsupportedError(f"unsupported constant type: {type(node.value)!r}")
    if isinstance(node, ast.Name):
        if node.id != var:
            raise ToolUnsupportedError(f"unknown symbol '{node.id}' without sympy")
        return {1: Fraction(1)}
    if isinstance(node, ast.BinOp):
        left = _poly_visit(node.left, var)
        right = _poly_visit(node.right, var)
        if isinstance(node.op, ast.Add):
            return poly_add(left, right)
        if isinstance(node.op, ast.Sub):
            return poly_add(left, right, sign=-1)
        if isinstance(node.op, ast.Mult):
            return poly_mul(left, right)
        if isinstance(node.op, ast.Div):
            if len(right) != 1 or 0 not in right:
                raise ToolUnsupportedError("division by a non-constant polynomial needs sympy")
            return poly_div_const(left, right[0])
        if isinstance(node.op, ast.Pow):
            if len(right) != 1 or 0 not in right:
                raise ToolUnsupportedError("non-integer polynomial exponents need sympy")
            exponent = right[0]
            if exponent.denominator != 1:
                raise ToolUnsupportedError("non-integer polynomial exponents need sympy")
            return poly_pow(left, int(exponent))
    if isinstance(node, ast.UnaryOp):
        inner = _poly_visit(node.operand, var)
        if isinstance(node.op, ast.UAdd):
            return inner
        if isinstance(node.op, ast.USub):
            return poly_mul({0: Fraction(-1)}, inner)
    raise ToolUnsupportedError(f"unsupported expression element: {ast.dump(node, include_attributes=False)}")


def poly_degree(poly: dict[int, Fraction]) -> int:
    return max(poly) if poly else -math.inf


def poly_roots(poly: dict[int, Fraction]) -> list[complex | float]:
    cleaned = _cleanup_poly(dict(poly))
    degree = poly_degree(cleaned)
    if degree == -math.inf:
        return []
    if degree == 0:
        return []
    if degree == 1:
        b = cleaned.get(1, Fraction(0))
        a = cleaned.get(0, Fraction(0))
        return [float(-a / b)]
    if degree == 2:
        a = cleaned.get(2, Fraction(0))
        b = cleaned.get(1, Fraction(0))
        c = cleaned.get(0, Fraction(0))
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            root = math.sqrt(float(discriminant))
        else:
            root = cmath.sqrt(complex(float(discriminant)))
        denom = 2 * float(a)
        return [(-float(b) - root) / denom, (-float(b) + root) / denom]
    raise ToolUnsupportedError("polynomial roots beyond degree 2 need sympy")


def solve_linear_system(matrix: list[list[Any]], rhs: list[Any]) -> list[float]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    if len(matrix) != len(rhs):
        raise ValueError("matrix and rhs dimensions do not match")

    size = len(matrix)
    augmented = [
        [_to_fraction(value) for value in row] + [_to_fraction(rhs[idx])]
        for idx, row in enumerate(matrix)
    ]

    for col in range(size):
        pivot = None
        for row in range(col, size):
            if augmented[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ToolUnsupportedError("system is singular or underdetermined")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for j in range(col, size + 1):
            augmented[col][j] /= pivot_value

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            for j in range(col, size + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [float(augmented[i][size]) for i in range(size)]


def determinant(matrix: list[list[Any]]) -> float:
    square = _validate_square_matrix(matrix)
    size = len(square)
    working = [[_to_fraction(value) for value in row] for row in square]
    sign = Fraction(1)

    for col in range(size):
        pivot = None
        for row in range(col, size):
            if working[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return 0.0
        if pivot != col:
            working[col], working[pivot] = working[pivot], working[col]
            sign *= -1
        pivot_value = working[col][col]
        for row in range(col + 1, size):
            factor = working[row][col] / pivot_value
            for j in range(col, size):
                working[row][j] -= factor * working[col][j]

    det = sign
    for idx in range(size):
        det *= working[idx][idx]
    return float(det)


def inverse(matrix: list[list[Any]]) -> list[list[float]]:
    square = _validate_square_matrix(matrix)
    size = len(square)
    augmented = [
        [_to_fraction(value) for value in row] + [Fraction(1 if i == j else 0) for j in range(size)]
        for i, row in enumerate(square)
    ]

    for col in range(size):
        pivot = None
        for row in range(col, size):
            if augmented[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ToolUnsupportedError("matrix is singular and cannot be inverted")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for j in range(2 * size):
            augmented[col][j] /= pivot_value

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            for j in range(2 * size):
                augmented[row][j] -= factor * augmented[col][j]

    return [[float(augmented[i][j]) for j in range(size, 2 * size)] for i in range(size)]


def rank(matrix: list[list[Any]]) -> int:
    if not matrix:
        return 0
    working = [[_to_fraction(value) for value in row] for row in matrix]
    rows = len(working)
    cols = len(working[0])
    pivot_row = 0
    rank_value = 0

    for col in range(cols):
        pivot = None
        for row in range(pivot_row, rows):
            if working[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != pivot_row:
            working[pivot_row], working[pivot] = working[pivot], working[pivot_row]
        pivot_value = working[pivot_row][col]
        for j in range(col, cols):
            working[pivot_row][j] /= pivot_value
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = working[row][col]
            if factor == 0:
                continue
            for j in range(col, cols):
                working[row][j] -= factor * working[pivot_row][j]
        pivot_row += 1
        rank_value += 1
        if pivot_row == rows:
            break
    return rank_value


def eigenpairs_2x2(matrix: list[list[Any]]) -> tuple[list[complex | float], list[list[complex | float]]]:
    square = _validate_square_matrix(matrix)
    if len(square) != 2:
        raise ToolUnsupportedError("eigenvalues without sympy are only supported for 2x2 matrices")
    a, b = float(square[0][0]), float(square[0][1])
    c, d = float(square[1][0]), float(square[1][1])
    trace = a + d
    det = a * d - b * c
    disc = trace * trace - 4 * det
    root = math.sqrt(disc) if disc >= 0 else cmath.sqrt(disc)
    l1 = (trace - root) / 2
    l2 = (trace + root) / 2
    eigenvalues = [l1, l2]
    eigenvectors = [_eigenvector_2x2(a, b, c, d, ev) for ev in eigenvalues]
    return eigenvalues, eigenvectors


def _eigenvector_2x2(a: float, b: float, c: float, d: float, eigenvalue: complex | float) -> list[complex | float]:
    eps = 1e-12
    if abs(b) > eps:
        return [1.0, (eigenvalue - a) / b]
    if abs(c) > eps:
        return [(eigenvalue - d) / c, 1.0]
    return [1.0, 0.0]


def _validate_square_matrix(matrix: list[list[Any]]) -> list[list[Any]]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have the same length")
    if len(matrix) != width:
        raise ValueError("matrix must be square")
    return matrix


def quadratic_critical_points(poly: dict[int, Fraction]) -> list[float]:
    derivative = poly_diff(poly, order=1)
    roots = poly_roots(derivative)
    out: list[float] = []
    for root in roots:
        if isinstance(root, complex):
            if abs(root.imag) > 1e-12:
                continue
            out.append(float(root.real))
        else:
            out.append(float(root))
    return sorted(out)


def taylor_series(poly: dict[int, Fraction], center: Any, order: int, var: str = "x") -> dict[str, Any]:
    if order < 0:
        raise ValueError("order must be non-negative")
    center_value = _to_fraction(center)
    coefficients: list[float] = []
    series_terms: list[str] = []
    for k in range(order + 1):
        derivative = poly_diff(poly, order=k)
        value = poly_eval(derivative, center_value)
        coeff = value / math.factorial(k)
        coefficients.append(float(coeff))
        if coeff == 0:
            continue
        if k == 0:
            term = f"{float(coeff)}"
        elif center_value == 0:
            if k == 1:
                term = f"{float(coeff)}*{var}"
            else:
                term = f"{float(coeff)}*{var}**{k}"
        elif k == 1:
            term = f"{float(coeff)}*({var} - {float(center_value)})"
        else:
            term = f"{float(coeff)}*({var} - {float(center_value)})**{k}"
        series_terms.append(term)
    return {
        "center": float(center_value),
        "order": order,
        "coefficients": coefficients,
        "series": " + ".join(series_terms) if series_terms else "0",
    }


def evaluate_expression_at_point(expression: str, var: str, point: Any) -> float:
    poly = poly_from_ast(expression, var=var)
    value = poly_eval(poly, point)
    return float(value)
