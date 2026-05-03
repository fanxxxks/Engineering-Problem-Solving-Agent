"""Fix broken JSON files using Mimo LLM to correct syntax errors.

Handles LaTeX backslash escapes (\sum, \frac etc.) and other common
JSON syntax issues in Chinese math question datasets.

Usage:
    python creator/fix_json.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MIMO_CONFIG = SCRIPT_DIR / "mimo.txt"

# Files to fix
TARGETS = [
    PROJECT_DIR / "data" / "calculus.json",
    PROJECT_DIR / "data" / "al.json",
    PROJECT_DIR / "data" / "ph.json",
    PROJECT_DIR / "data" / "calculus" / "test.json",
    PROJECT_DIR / "验证集" / "基础物理学.json",
    PROJECT_DIR / "验证集" / "微积分.json",
    PROJECT_DIR / "验证集" / "线性代数题目.json",
]

# Files to skip (already working)
SKIP = {"testimage.json", "电路原理题集.json"}


# ── Mimo client ───────────────────────────────────────────────────────────

def _load_mimo_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    raw = MIMO_CONFIG.read_text("utf-8").strip()
    for line in raw.split("\n"):
        line = line.strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        env_match = re.match(r'os\.environ\.get\(["\'](.+?)["\']\)', val)
        if env_match:
            val = env_match.group(1)
        cfg[key] = val
    return cfg


def _build_client() -> OpenAI:
    import httpx
    cfg = _load_mimo_config()
    return OpenAI(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", ""),
        timeout=180.0,
        max_retries=1,
        http_client=httpx.Client(verify=False),
    )


# ── Mechanical fixes (fast, no LLM needed) ────────────────────────────────

def _try_mechanical_fix(raw: bytes) -> tuple[str, bool]:
    """Try to fix common JSON issues without calling an LLM.

    Returns (fixed_text, success).
    """
    # Decode: try multiple encodings
    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw.decode("utf-8-sig")
    else:
        for enc in ("utf-8", "gbk", "gb18030", "utf-8"):
            try:
                text = raw.decode(enc, errors="strict")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = raw.decode("utf-8", errors="replace")

    # Test if already valid
    try:
        json.loads(text)
        return text, True
    except json.JSONDecodeError:
        pass

    # Fix 1: LaTeX backslash escapes inside JSON strings
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    # Invalid: \s (in \sum), \f (in \frac), \i (in \int), \l (in \lim), etc.
    def _fix_latex(s: str) -> str:
        """Double all backslashes in JSON string values except \\\" and \\\\.

        JSON only allows \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX as escapes.
        LaTeX commands like \\sum \\frac \\int \\lim \\infty \\sin are all invalid.
        Even \\f in \\frac must be doubled because JSON would parse it as form-feed.
        """
        result = []
        in_string = False
        i = 0
        while i < len(s):
            ch = s[i]
            if in_string:
                if ch == "\\":
                    next_ch = s[i + 1] if i + 1 < len(s) else ""
                    if next_ch in ('"', '\\'):
                        result.append(ch)       # keep \" and \\ as-is
                    else:
                        result.append("\\\\")   # double everything else
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                result.append(ch)
            i += 1
        return "".join(result)

    fixed = _fix_latex(text)
    try:
        json.loads(fixed)
        return fixed, True
    except json.JSONDecodeError:
        return text, False


# ── LLM-based fix (for stubborn errors) ───────────────────────────────────

def _llm_fix(client: OpenAI, model: str, text: str, filename: str) -> str | None:
    """Ask Mimo to fix remaining JSON syntax errors."""
    # Send in 100KB chunks for very large files
    if len(text) > 100000:
        return _llm_fix_chunked(client, model, text, filename)

    prompt = (
        "你是一个 JSON 修复工具。下面是一段有语法错误的 JSON 文本。\n"
        "请修复所有 JSON 语法错误（如未转义的反斜杠、多余的逗号、缺失的引号等）\n"
        "并直接返回完整的、合法的 JSON 文本。不要添加任何解释，只返回 JSON。\n\n"
        f"{text}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        # Extract just the JSON part
        start = content.find("[") if "[" in content else content.find("{")
        end = content.rfind("]") if "]" in content else content.rfind("}")
        if start >= 0 and end > start:
            fixed = content[start : end + 1]
        else:
            fixed = content
        # Validate
        json.loads(fixed)
        return fixed
    except Exception as exc:
        print(f"      LLM fix failed: {exc}")
        return None


def _llm_fix_chunked(client: OpenAI, model: str, text: str, filename: str) -> str | None:
    """Fix a large JSON file by processing one JSON object at a time."""
    # Split into individual objects by scanning braces
    objects = []
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
                objects.append(text[start : i + 1])
                start = -1

    if not objects:
        return None

    fixed_objects = []
    for idx, obj_text in enumerate(objects):
        # Quick check: is this object already valid JSON?
        try:
            json.loads(obj_text)
            fixed_objects.append(obj_text)
            continue
        except json.JSONDecodeError:
            pass

        # Send to LLM for fixing
        prompt = (
            "你是一个 JSON 修复工具。请修复下面这个 JSON 对象的语法错误，\n"
            "返回修复后的完整 JSON 对象，不要加任何解释文字。\n\n"
            f"{obj_text}"
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
            )
            content = response.choices[0].message.content
            if content:
                start = content.find("{")
                end = content.rfind("}")
                if start >= 0 and end > start:
                    fixed = content[start : end + 1]
                    json.loads(fixed)  # validate
                    fixed_objects.append(fixed)
                    if (idx + 1) % 20 == 0:
                        print(f"      {idx+1}/{len(objects)} objects fixed...")
                    continue
        except Exception:
            pass
        fixed_objects.append(obj_text)  # keep original if can't fix

    result = "[\n" + ",\n".join(fixed_objects) + "\n]"
    try:
        json.loads(result)
        return result
    except json.JSONDecodeError:
        return None


# ── Main ───────────────────────────────────────────────────────────────────

def fix_file(path: Path, client: OpenAI, model: str) -> bool:
    """Fix a single JSON file. Returns True if now valid."""
    print(f"\n{'='*60}")
    print(f"Fixing: {path.name} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"{'='*60}")

    raw = path.read_bytes()

    # Step 1: Try mechanical fix
    text, ok = _try_mechanical_fix(raw)
    if ok:
        path.write_text(text, encoding="utf-8")
        data = json.loads(text)
        n = len(data) if isinstance(data, list) else len(data.get("questions", []))
        print(f"  [OK] Mechanical fix succeeded ({n} items)")
        return True

    # Step 2: Try LLM fix
    print(f"  Mechanical fix failed, asking Mimo...")
    fixed = _llm_fix(client, model, text, path.name)
    if fixed:
        path.write_text(fixed, encoding="utf-8")
        data = json.loads(fixed)
        n = len(data) if isinstance(data, list) else len(data.get("questions", []))
        print(f"  [OK] Mimo fix succeeded ({n} items)")
        return True

    print(f"  [FAIL] Could not fix {path.name}")
    return False


def main() -> int:
    cfg = _load_mimo_config()
    model = cfg.get("model", "mimo-v2.5-pro")
    client = _build_client()

    ok_count = 0
    fail_count = 0

    for path in TARGETS:
        if not path.exists():
            print(f"  [SKIP] Not found: {path}")
            continue
        if fix_file(path, client, model):
            ok_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Done: {ok_count} fixed, {fail_count} failed")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
