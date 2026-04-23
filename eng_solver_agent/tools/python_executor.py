"""Python code execution utility for tool-assisted reasoning."""

from __future__ import annotations

import contextlib
import io


def execute_python_code(code: str) -> str:
    prefix = "import sympy as sp\nimport numpy as np\nimport math\n"
    full_code = prefix + str(code)
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(full_code, {})
        output = stdout.getvalue().strip()
        if output:
            return output
        return "代码执行成功，但没有打印(print)任何输出。请确保你在代码中 print 了结果。"
    except Exception as exc:
        return f"代码执行报错:\n{exc}"
