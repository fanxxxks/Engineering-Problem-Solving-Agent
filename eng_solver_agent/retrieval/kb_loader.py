"""Local knowledge-base loader for formula cards and solved examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KnowledgeBaseLoader:
    """Loads local reference data from JSON or JSONL files."""

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        candidate = Path(path)
        if not candidate.exists():
            return []
        if candidate.suffix.lower() == ".jsonl":
            return self._load_jsonl(candidate)
        if candidate.suffix.lower() == ".json":
            return self._load_json(candidate)
        raise ValueError(f"unsupported knowledge-base file type: {candidate.suffix}")

    def load_formula_cards(self, path: str | Path) -> list[dict[str, Any]]:
        return self.load(path)

    def load_solved_examples(self, path: str | Path) -> list[dict[str, Any]]:
        return self.load(path)

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to load JSON knowledge base: {path}") from exc
        if isinstance(data, list):
            return [self._ensure_dict(item, path) for item in data]
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                return [self._ensure_dict(item, path) for item in data["items"]]
            return [self._ensure_dict(data, path)]
        raise ValueError(f"JSON knowledge base must contain an object or array: {path}")

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except Exception as exc:
                        raise ValueError(f"failed to parse JSONL line {line_number} in {path}") from exc
                    records.append(self._ensure_dict(record, path))
        except OSError as exc:
            raise ValueError(f"failed to read knowledge base: {path}") from exc
        return records

    def _ensure_dict(self, item: Any, path: Path) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"knowledge-base record must be an object in {path}")
        return item
