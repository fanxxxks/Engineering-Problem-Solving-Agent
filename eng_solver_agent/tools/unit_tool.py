"""Lightweight unit/dimension checker for reasoning text."""

from __future__ import annotations

import re
from typing import Any


class UnitTool:
    _DIMENSIONS: dict[str, dict[str, int]] = {
        "m": {"L": 1},
        "s": {"T": 1},
        "kg": {"M": 1},
        "A": {"I": 1},
        "V": {"M": 1, "L": 2, "T": -3, "I": -1},
        "Ω": {"M": 1, "L": 2, "T": -3, "I": -2},
        "N": {"M": 1, "L": 1, "T": -2},
        "J": {"M": 1, "L": 2, "T": -2},
        "W": {"M": 1, "L": 2, "T": -3},
        "C": {"T": 1, "I": 1},
        "Hz": {"T": -1},
    }

    def normalize(self, value: str) -> str:
        return value.strip()

    def dimension_of(self, unit: str) -> dict[str, int]:
        normalized = self.normalize(unit)
        if normalized not in self._DIMENSIONS:
            raise ValueError(f"unknown unit: {unit}")
        return dict(self._DIMENSIONS[normalized])

    def compatible(self, unit_a: str, unit_b: str) -> bool:
        return self.dimension_of(unit_a) == self.dimension_of(unit_b)

    def check_reasoning_process(self, reasoning_process: str) -> dict[str, Any]:
        found_units = self._extract_units(reasoning_process)
        unknown_tokens = self._extract_unknown_unit_tokens(reasoning_process, found_units)
        return {
            "recognized_units": found_units,
            "unknown_tokens": unknown_tokens,
            "has_known_units": bool(found_units),
        }

    def _extract_units(self, text: str) -> list[str]:
        pattern = re.compile(r"\b(?:kg|Hz|Ω|m|s|A|V|N|J|W|C)\b")
        hits = pattern.findall(text)
        return sorted(set(self.normalize(hit) for hit in hits))

    def _extract_unknown_unit_tokens(self, text: str, known_units: list[str]) -> list[str]:
        token_pattern = re.compile(r"\b[^\W\d_]+\b", re.UNICODE)
        known = set(known_units)
        candidates = []
        for token in token_pattern.findall(text):
            if token in known:
                continue
            if token in self._DIMENSIONS:
                continue
            if len(token) <= 1:
                continue
            candidates.append(token)
        return sorted(set(candidates))
