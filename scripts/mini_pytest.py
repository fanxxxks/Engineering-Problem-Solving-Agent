"""Tiny pytest-compatible runner for this scaffold's local tests."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from types import ModuleType


def load_module(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_test_functions(module: ModuleType):
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            yield name, obj


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "-q" in argv or "--quiet" in argv
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    test_files = sorted((root / "tests").glob("test_*.py"))
    passed = 0
    failed = 0

    for test_file in test_files:
        try:
            module = load_module(test_file)
        except Exception:
            failed += 1
            if quiet:
                print("E", end="", flush=True)
            else:
                print(f"ERROR {test_file.name}")
                traceback.print_exc()
            continue

        for test_name, test_func in iter_test_functions(module):
            try:
                test_func()
                passed += 1
                if quiet:
                    print(".", end="", flush=True)
                else:
                    print(f"PASS {test_file.name}::{test_name}")
            except Exception:
                failed += 1
                if quiet:
                    print("F", end="", flush=True)
                else:
                    print(f"FAIL {test_file.name}::{test_name}")
                traceback.print_exc()

    if quiet:
        print()
    print(f"{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
