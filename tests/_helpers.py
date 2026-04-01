"""Small test helpers used by the local mini test runner."""

from __future__ import annotations

from typing import Any, Callable, Type


def assert_raises(expected_exception: Type[BaseException], func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        func(*args, **kwargs)
    except expected_exception:
        return
    except Exception as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(f"expected {expected_exception.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {expected_exception.__name__} to be raised")


def approx_equal(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance
