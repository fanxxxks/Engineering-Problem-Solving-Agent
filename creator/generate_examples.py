"""Generate solved-example cards from question-answer JSON files.

Reads all question+answer pairs from the validation set and data directory,
asks the Mimo LLM to produce a detailed reasoning_process, and writes a
JSONL file compatible with the existing solved_examples.jsonl knowledge base.

Usage:
    python creator/generate_examples.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MIMO_CONFIG = SCRIPT_DIR / "mimo.txt"
OUTPUT_FILE = PROJECT_DIR / "eng_solver_agent" / "retrieval" / "solved_examples.jsonl"

# Existing question IDs to avoid duplicates
_EXISTING_IDS: set[str] = set()

# Subject keyword mapping for topic inference
_SUBJECT_HINTS: dict[str, list[str]] = {
    "calculus": ["导数", "积分", "极限", "微分", "derivative", "integral", "limit", "taylor", "级数", "收敛"],
    "linalg":   ["矩阵", "行列式", "特征值", "特征向量", "逆矩阵", "秩", "线性", "matrix", "determinant", "eigen"],
    "circuits": ["电路", "电阻", "电容", "电感", "电压", "电流", "Ω", "KCL", "KVL", "等效", "串联", "并联"],
    "physics":  ["力", "速度", "加速度", "动量", "能量", "功", "牛顿", "kg", "m/s", "N"],
}

_TOPIC_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "calculus": [
        ("derivative", ["导数", "求导", "derivative", "differentiate"]),
        ("integration", ["积分", "integral", "integrate"]),
        ("limit", ["极限", "limit"]),
        ("taylor_series", ["泰勒", "taylor", "级数"]),
    ],
    "linalg": [
        ("determinant", ["行列式", "determinant"]),
        ("matrix_inverse", ["逆矩阵", "inverse"]),
        ("eigenvalues", ["特征值", "特征向量", "eigen"]),
        ("rank", ["秩", "rank"]),
        ("matrix_power", ["幂", "power"]),
    ],
    "circuits": [
        ("equivalent_resistance", ["等效电阻", "串联", "并联", "equivalent resistance"]),
        ("ohms_law", ["欧姆", "ohm", "伏安"]),
        ("basic_circuits", ["电路", "KCL", "KVL"]),
    ],
    "physics": [
        ("newton_second_law", ["牛顿第二", "F=ma", "newton"]),
        ("uniform_acceleration", ["匀加速", "匀变速", "加速度"]),
        ("momentum", ["动量", "momentum", "冲量"]),
        ("work_energy", ["功", "能量", "动能", "work", "energy"]),
    ],
}


# ── Mimo client ───────────────────────────────────────────────────────────

def _load_mimo_config() -> dict[str, str]:
    """Parse mimo.txt for api_key, base_url, model."""
    cfg: dict[str, str] = {}
    raw = MIMO_CONFIG.read_text("utf-8").strip()
    for line in raw.split("\n"):
        line = line.strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Handle os.environ.get("KEY") pattern — KEY is the actual api key
        env_match = re.match(r'os\.environ\.get\(["\'](.+?)["\']\)', val)
        if env_match:
            val = env_match.group(1)  # the string inside quotes IS the key
        cfg[key] = val
    return cfg


def _build_client() -> OpenAI:
    cfg = _load_mimo_config()
    return OpenAI(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", ""),
        timeout=180.0,
        max_retries=2,
    )


# ── JSON loading with error recovery ──────────────────────────────────────

def _load_json_robust(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file, attempting multiple encodings and error recovery."""
    raw = path.read_bytes()

    # Try UTF-16
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig")
    else:
        text = raw.decode("utf-8", errors="replace")

    # Try standard parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
    except json.JSONDecodeError:
        pass

    # Recovery: try to extract individual JSON objects from the text
    return _extract_objects_bruteforce(text)


def _extract_objects_bruteforce(text: str) -> list[dict[str, Any]]:
    """Extract valid JSON objects from corrupted text by scanning braces."""
    objects: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict) and "question" in obj:
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return objects


def _collect_questions() -> list[dict[str, Any]]:
    """Gather all question+answer pairs from the project."""
    all_qs: list[dict[str, Any]] = []
    search_dirs = [
        PROJECT_DIR / "验证集",
        PROJECT_DIR / "data" / "dev",
        PROJECT_DIR / "data",
    ]

    SKIP_FILES = {"testimage.json", "电路原理题集.json"}

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for json_file in sorted(search_dir.glob("*.json")):
            if json_file.name in SKIP_FILES:
                continue
            try:
                items = _load_json_robust(json_file)
            except Exception:
                print(f"  [SKIP] Cannot parse: {json_file.name}")
                continue
            valid = [q for q in items if isinstance(q, dict) and q.get("question") and q.get("answer")]
            if valid:
                all_qs.extend(valid)
                print(f"  {json_file.name}: {len(valid)} questions")
    return all_qs


# ── Topic inference ───────────────────────────────────────────────────────

def _infer_subject(question: dict[str, Any]) -> str:
    """Infer subject from question fields or text."""
    existing = str(question.get("subject", "")).strip().lower()
    if existing in ("physics", "circuits", "linalg", "calculus"):
        return existing
    qid = str(question.get("question_id", "")).lower()
    for sub in ("physics", "circuits", "linalg", "calculus"):
        if sub in qid:
            return sub
    text = str(question.get("question", "")).lower()
    for sub, keywords in _SUBJECT_HINTS.items():
        if any(k in text for k in keywords):
            return sub
    return "calculus"


def _infer_topic(subject: str, question: dict[str, Any]) -> str:
    """Infer topic from question text."""
    existing = str(question.get("topic", "")).strip()
    if existing and existing != "unknown":
        return existing
    text = str(question.get("question", "")).lower()
    patterns = _TOPIC_PATTERNS.get(subject, [])
    for topic, keywords in patterns:
        if any(k in text for k in keywords):
            return topic
    return subject  # fallback


def _infer_tags(subject: str, topic: str, question: dict[str, Any]) -> list[str]:
    """Infer tags from question text."""
    text = str(question.get("question", ""))
    existing = question.get("tags", [])
    if isinstance(existing, list) and existing:
        return existing
    tags = [subject, topic]
    all_keywords = _SUBJECT_HINTS.get(subject, [])
    for kw in all_keywords:
        if kw in text.lower():
            tags.append(kw)
    return list(set(tags))[:6]


# ── Reasoning generation ──────────────────────────────────────────────────

def _build_reasoning_prompt(question: dict[str, Any], subject: str, topic: str) -> list[dict[str, Any]]:
    """Build the prompt for generating reasoning."""
    q_text = question.get("question", "")
    answer = question.get("answer", "")

    system = (
        "你是一位工科解题专家。请为下面的题目生成详细的解题步骤（reasoning_process），"
        "使用中文，格式为：已知条件 → 所用公式/定理 → 代入计算 → 结论。"
        "只返回 JSON，格式：\n"
        '{"reasoning_process": "解题步骤...", "topic": "知识点名称", "tags": ["标签1", "标签2"]}'
    )
    user = (
        f"题目：{q_text}\n"
        f"答案：{answer}\n"
        f"学科：{subject}\n\n"
        f"请生成解题步骤。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _generate_reasoning(
    client: OpenAI, model: str, question: dict[str, Any], subject: str, topic: str
) -> dict[str, Any] | None:
    """Call Mimo to generate reasoning_process."""
    messages = _build_reasoning_prompt(question, subject, topic)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        # Extract JSON from response
        cleaned = _extract_json(content)
        data = json.loads(cleaned)
        return data
    except Exception as exc:
        print(f"    API error: {exc}")
        return None


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (may contain markdown fences)."""
    for pat in [r"```json\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                json.loads(m.group(1))
                return m.group(1)
            except json.JSONDecodeError:
                continue
    # Try to find first { } block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    # Load existing IDs
    if OUTPUT_FILE.exists():
        for line in OUTPUT_FILE.read_text("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                _EXISTING_IDS.add(ex.get("question_id", ""))
            except json.JSONDecodeError:
                pass
        print(f"Existing examples: {len(_EXISTING_IDS)}")

    # Collect questions
    print("\nLoading questions...")
    questions = _collect_questions()
    print(f"Total: {len(questions)} questions")

    # Filter out already-processed
    pending = [q for q in questions if q.get("question_id") not in _EXISTING_IDS]
    print(f"Pending: {len(pending)} (skipping {len(questions) - len(pending)} already done)\n")

    # Build clients (one per thread)
    cfg = _load_mimo_config()
    model = cfg.get("model", "mimo-v2.5-pro")
    MAX_WORKERS = 8
    write_lock = Lock()
    stats = {"generated": 0, "failed": 0}

    def _process_one(q: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        qid = q.get("question_id", "?")
        subject = _infer_subject(q)
        topic = _infer_topic(subject, q)
        client = _build_client()
        result = _generate_reasoning(client, model, q, subject, topic)
        if result and result.get("reasoning_process"):
            card = {
                "question_id": qid,
                "subject": subject,
                "topic": result.get("topic", topic),
                "question": q.get("question", ""),
                "reasoning_process": result["reasoning_process"],
                "answer": q.get("answer", ""),
                "tags": result.get("tags", _infer_tags(subject, topic, q)),
            }
            return card, qid
        return None, qid

    with OUTPUT_FILE.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, q): q for q in pending}
            for i, future in enumerate(as_completed(futures)):
                q = futures[future]
                qid = q.get("question_id", "?")
                try:
                    card, _ = future.result()
                except Exception as exc:
                    card = None
                    print(f"[{i+1}/{len(pending)}] {qid} ERROR: {exc}")

                if card:
                    with write_lock:
                        out_fh.write(json.dumps(card, ensure_ascii=False) + "\n")
                        out_fh.flush()
                    stats["generated"] += 1
                    print(f"[{i+1}/{len(pending)}] {qid} OK ({card['subject']}/{card['topic']})")
                else:
                    stats["failed"] += 1
                    print(f"[{i+1}/{len(pending)}] {qid} FAILED")

    print(f"\nDone: {stats['generated']} generated, {stats['failed']} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
