"""Physics tool layer with a compact structured solver entrypoint."""

from __future__ import annotations

import math
from typing import Any

from eng_solver_agent.tools._math_support import ToolUnsupportedError, load_sympy


class PhysicsTool:
    def __init__(self) -> None:
        self._sympy = load_sympy()

    def solve(self, expression: str) -> str:
        return f"[physics-tool] {expression}"

    def solve_relation(self, relation: str, knowns: dict[str, Any], target: str) -> float:
        normalized = relation.strip().lower()
        target_raw = target.strip()
        target_lower = target_raw.lower()
        if normalized in {"uniform_acceleration", "kinematics", "constant_acceleration"}:
            return self._solve_uniform_acceleration(knowns, target_raw, target_lower)
        if normalized in {"newton_second_law", "newton2", "f_ma"}:
            return self._solve_newton_second_law(knowns, target_raw, target_lower)
        if normalized in {"work_energy", "energy"}:
            return self._solve_work_energy(knowns, target_raw, target_lower)
        if normalized in {"momentum", "impulse"}:
            return self._solve_momentum(knowns, target_raw, target_lower)
        raise ToolUnsupportedError(f"unsupported physics relation: {relation}")

    def _solve_uniform_acceleration(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        v0 = knowns.get("v0")
        v = knowns.get("v")
        a = knowns.get("a")
        t = knowns.get("t")
        s = knowns.get("s")

        if target in {"v", "final_velocity"}:
            if v0 is None or a is None or t is None:
                raise ValueError("uniform_acceleration target 'v' requires v0, a, and t")
            return float(v0 + a * t)
        if target in {"s", "x", "displacement"}:
            if v0 is not None and a is not None and t is not None:
                return float(v0 * t + 0.5 * a * t * t)
            if v is not None and v0 is not None and a is not None:
                return float((v * v - v0 * v0) / (2 * a))
            raise ValueError("uniform_acceleration target 's' requires either (v0, a, t) or (v, v0, a)")
        if target in {"a", "acceleration"}:
            if v is None or v0 is None or t is None:
                raise ValueError("uniform_acceleration target 'a' requires v, v0, and t")
            return float((v - v0) / t)
        if target in {"t", "time"}:
            if v is not None and v0 is not None and a is not None:
                if a == 0:
                    raise ValueError("acceleration cannot be zero when solving for time")
                return float((v - v0) / a)
            if s is not None and v0 is not None and a is not None:
                return float(self._solve_quadratic_time(a, v0, s))
            raise ValueError("uniform_acceleration target 't' requires either (v, v0, a) or (s, v0, a)")
        raise ToolUnsupportedError(f"unsupported kinematics target: {target_raw}")

    def _solve_quadratic_time(self, a: Any, v0: Any, s: Any) -> float:
        if a == 0:
            if v0 == 0:
                raise ValueError("insufficient information to solve for time")
            return float(s / v0)
        discriminant = v0 * v0 + 2 * a * s
        if discriminant < 0:
            raise ValueError("no real solution for time")
        root = math.sqrt(float(discriminant))
        t1 = (-float(v0) + root) / float(a)
        t2 = (-float(v0) - root) / float(a)
        candidates = [value for value in (t1, t2) if value >= 0]
        if not candidates:
            raise ValueError("no non-negative time solution")
        return min(candidates)

    def _solve_newton_second_law(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        F = knowns.get("F")
        m = knowns.get("m")
        a = knowns.get("a")

        if target in {"f"}:
            if m is None or a is None:
                raise ValueError("Newton's second law target 'F' requires m and a")
            return float(m * a)
        if target == "m":
            if F is None or a is None:
                raise ValueError("Newton's second law target 'm' requires F and a")
            if a == 0:
                raise ValueError("acceleration cannot be zero when solving for mass")
            return float(F / a)
        if target == "a":
            if F is None or m is None:
                raise ValueError("Newton's second law target 'a' requires F and m")
            if m == 0:
                raise ValueError("mass cannot be zero when solving for acceleration")
            return float(F / m)
        raise ToolUnsupportedError(f"unsupported Newton target: {target_raw}")

    def _solve_work_energy(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        m = knowns.get("m")
        v0 = knowns.get("v0")
        v = knowns.get("v")
        F = knowns.get("F")
        s = knowns.get("s")
        theta = knowns.get("theta", 0)
        theta_rad = math.radians(float(theta))

        if target in {"w"}:
            if m is not None and v0 is not None and v is not None:
                return float(0.5 * m * (v * v - v0 * v0))
            if F is not None and s is not None:
                return float(F * s * math.cos(theta_rad))
            raise ValueError("work-energy target 'W' requires either (m, v0, v) or (F, s, theta)")
        if target in {"f"}:
            if s is None:
                raise ValueError("work-energy target 'F' requires s")
            work = knowns.get("W")
            if work is None:
                raise ValueError("work-energy target 'F' requires W")
            if s == 0:
                raise ValueError("distance cannot be zero when solving for force")
            return float(work / (s * math.cos(theta_rad)))
        raise ToolUnsupportedError(f"unsupported work-energy target: {target_raw}")

    def _solve_momentum(self, knowns: dict[str, Any], target_raw: str, target: str) -> float:
        m = knowns.get("m")
        v = knowns.get("v")
        p = knowns.get("p")
        F = knowns.get("F")
        dt = knowns.get("dt")

        if target == "p":
            if m is None or v is None:
                raise ValueError("momentum target 'p' requires m and v")
            return float(m * v)
        if target == "v":
            if p is None or m is None:
                raise ValueError("momentum target 'v' requires p and m")
            if m == 0:
                raise ValueError("mass cannot be zero when solving for velocity")
            return float(p / m)
        if target == "impulse":
            if F is None or dt is None:
                raise ValueError("momentum target 'impulse' requires F and dt")
            return float(F * dt)
        raise ToolUnsupportedError(f"unsupported momentum target: {target_raw}")
