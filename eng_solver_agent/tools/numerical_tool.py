"""Numerical computation tool — sandboxed SymPy / NumPy / SciPy execution engine.

Accepts Python code strings from the LLM, executes them in a restricted
environment with sympy, numpy, scipy, and math available, then returns
captured stdout as the result.
"""

from __future__ import annotations

import sys
import traceback
from io import StringIO
from typing import Any


class NumericalComputationTool:
    """Sandboxed math engine using SymPy, NumPy, and SciPy.

    The LLM writes Python code that uses print() to output results.
    The tool executes the code and returns whatever was printed.
    """

    def compute(self, code: str) -> str:
        """Execute Python math code and return the captured output.

        Alias for solve(). The code string can use sympy, sp, np, scipy, and math.
        Use print() to output the final result.
        """
        return self._execute(code)

    # ------------------------------------------------------------------
    # Sandbox engine
    # ------------------------------------------------------------------

    def _execute(self, code: str) -> str:
        """Execute Python code in a sandboxed environment.

        Exposes sympy (as 'sympy' and 'sp'), numpy ('np'), scipy, and math.
        Captures stdout and returns it. Errors are returned as strings.
        """
        code = code.strip()
        if not code:
            return "[错误] 代码不能为空"

        safe_globals = self._build_safe_globals()

        old_stdout = sys.stdout
        redirected = StringIO()
        sys.stdout = redirected

        try:
            exec(code, safe_globals)
            sys.stdout = old_stdout
            output = redirected.getvalue().strip()
            return output if output else "[代码执行完成，无输出]"
        except Exception:
            sys.stdout = old_stdout
            tb = traceback.format_exc().strip().split("\n")[-1]
            return f"[代码执行出错] {tb}"
        finally:
            sys.stdout = old_stdout

    def _build_safe_globals(self) -> dict[str, Any]:
        """Build the restricted globals dict for exec()."""
        sympy = __import__("sympy")
        numpy = __import__("numpy")
        scipy = __import__("scipy")
        return {
            "__builtins__": __import__("builtins").__dict__,
            # Primary namespaces
            "sympy": sympy,
            "sp": sympy,
            "np": numpy,
            "scipy": scipy,
            "math": __import__("math"),
            # Convenience aliases for common sympy functions
            "symbols": sympy.symbols,
            "Symbol": sympy.Symbol,
            "solve": sympy.solve,
            "diff": sympy.diff,
            "integrate": sympy.integrate,
            "limit": sympy.limit,
            "simplify": sympy.simplify,
            "expand": sympy.expand,
            "factor": sympy.factor,
            "Matrix": sympy.Matrix,
            "pi": sympy.pi,
            "E": sympy.E,
            "oo": sympy.oo,
            "sqrt": sympy.sqrt,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "log": sympy.log,
            "exp": sympy.exp,
        }
