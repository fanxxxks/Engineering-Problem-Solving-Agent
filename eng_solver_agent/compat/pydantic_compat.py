"""Minimal Pydantic-compatible subset for this project.

The current machine does not have the real `pydantic` package installed, so
this internal shim keeps the repository runnable. If the environment supports
it later, prefer switching imports back to the real package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


class ValidationError(ValueError):
    """Raised when model validation fails."""


_MISSING = object()


@dataclass(frozen=True)
class _FieldInfo:
    default: Any = _MISSING
    default_factory: Callable[[], Any] | object = _MISSING


def Field(default: Any = _MISSING, *, default_factory: Callable[[], Any] | object = _MISSING):
    return _FieldInfo(default=default, default_factory=default_factory)


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) in get_args(annotation)
    return False


def _matches(annotation: Any, value: Any) -> bool:
    if annotation is Any or annotation is object:
        return True
    if value is None:
        return _is_optional(annotation)

    origin = get_origin(annotation)
    if origin is Union:
        return any(_matches(arg, value) for arg in get_args(annotation) if arg is not type(None))
    if origin in (list, list[str], list[Any]):
        return isinstance(value, list)
    if origin in (dict, dict[str, Any]):
        return isinstance(value, dict)
    if str(origin) == "typing.Literal":
        return value in get_args(annotation)
    if annotation is str:
        return isinstance(value, str)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is bool:
        return isinstance(value, bool)
    return isinstance(value, annotation)


class BaseModel:
    """Tiny subset of `pydantic.BaseModel`."""

    def __init__(self, **data: Any) -> None:
        annotations = get_type_hints(type(self))
        values: dict[str, Any] = {}
        for name, annotation in annotations.items():
            if name.startswith("_"):
                continue
            if name in data:
                value = data[name]
            else:
                default = getattr(type(self), name, _MISSING)
                if isinstance(default, _FieldInfo):
                    if default.default_factory is not _MISSING:
                        value = default.default_factory()  # type: ignore[misc]
                    elif default.default is not _MISSING:
                        value = default.default
                    else:
                        raise ValidationError(f"Missing required field: {name}")
                elif default is not _MISSING:
                    value = default
                elif _is_optional(annotation):
                    value = None
                else:
                    raise ValidationError(f"Missing required field: {name}")

            if not _matches(annotation, value):
                raise ValidationError(
                    f"Invalid value for field '{name}': expected {annotation!r}, got {type(value)!r}"
                )
            values[name] = value

        extra = set(data) - set(annotations)
        if extra:
            for name in extra:
                values[name] = data[name]

        object.__setattr__(self, "__dict__", values)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def model_validate(cls, data: Any) -> "BaseModel":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValidationError(f"Expected dict for model validation, got {type(data)!r}")
        return cls(**data)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({args})"
