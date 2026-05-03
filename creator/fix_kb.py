"""Fix knowledge base: verify subject/topic, dedup, regenerate IDs via Mimo.

Usage:
    python creator/fix_kb.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MIMO_CONFIG = SCRIPT_DIR / "mimo.txt"
KB_FILE = PROJECT_DIR / "eng_solver_agent" / "retrieval" / "solved_examples.jsonl"
OUTPUT_FILE = PROJECT_DIR / "eng_solver_agent" / "retrieval" / "solved_examples_fixed.jsonl"

MAX_WORKERS = 8
BATCH_SIZE = 20


def _load_mimo_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in MIMO_CONFIG.read_text("utf-8").strip().split("\n"):
        if "=" not in line: continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("\"'")
        m = re.match(r'os\.environ\.get\(["\'](.+?)["\']\)', v)
        cfg[k] = m.group(1) if m else v
    return cfg


def _build_client() -> OpenAI:
    import httpx
    cfg = _load_mimo_config()
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=120.0, max_retries=1,
                  http_client=httpx.Client(verify=False))


_FIX_SYSTEM = (
    "你是一个知识库审核专家。下面是一批知识卡片（JSON数组），请逐一检查并修正：\n"
    "1. subject 必须是 physics / circuits / linalg / calculus 之一，根据题目内容判断\n"
    "2. topic 必须是准确的子主题（如 derivative / limit / newton_second_law 等）\n"
    "3. question_id 用格式：{SUBJECT}_{TOPIC}_{序号}，如 CALC_DERIV_042\n"
    "4. 如果原题有明显错误（subject不对、topic不对），修正它们\n"
    "5. 不要修改 question / reasoning_process / answer 的内容\n"
    "只返回修正后的 JSON 数组。"
)


def _fix_batch(client: OpenAI, model: str, cards: list[dict]) -> list[dict]:
    """Send a batch of cards to Mimo for fixing."""
    inp = json.dumps(cards, ensure_ascii=False, indent=1)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FIX_SYSTEM},
                {"role": "user", "content": inp},
            ],
            temperature=1.0,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"    API error: {exc}")
        return cards

    # Extract JSON array
    for pat in [r"```json\s*\n?(.*?)\n?```", r"```\s*\n?(.*?)\n?```"]:
        m = re.search(pat, raw, re.DOTALL)
        if m: raw = m.group(1).strip(); break
    start = raw.find("["); end = raw.rfind("]")
    if start >= 0 and end > start: raw = raw[start:end+1]
    try:
        fixed = json.loads(raw)
        if isinstance(fixed, list):
            return fixed
    except json.JSONDecodeError:
        pass
    return cards


def main() -> int:
    if not KB_FILE.exists():
        print("No knowledge base found!"); return 1

    cards = []
    for line in KB_FILE.read_text("utf-8").strip().split("\n"):
        if not line.strip(): continue
        try: cards.append(json.loads(line))
        except json.JSONDecodeError: pass

    print(f"Loaded {len(cards)} cards")
    cfg = _load_mimo_config()
    model = cfg["model"]

    fixed_cards: list[dict] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    total = len(cards)
    fixed_count = 0

    def _hash(q): return __import__('hashlib').sha256(
        " ".join(str(q)[:300].lower().split()).encode()).hexdigest()[:16]

    for batch_start in range(0, total, BATCH_SIZE):
        batch = cards[batch_start: batch_start + BATCH_SIZE]
        print(f"\nBatch {batch_start//BATCH_SIZE + 1}/{(total+BATCH_SIZE-1)//BATCH_SIZE} ({len(batch)} cards) → fixing...")
        result = _fix_batch(_build_client(), model, batch)

        for card in result:
            qid = card.get("question_id", "")
            qhash = _hash(card.get("question", ""))
            if qhash in seen_hashes:
                continue  # dedup
            # Ensure unique ID
            if qid in seen_ids:
                base = re.sub(r'_\d+$', '', qid)
                n = 1
                while f"{base}_{n:03d}" in seen_ids: n += 1
                card["question_id"] = f"{base}_{n:03d}"
            seen_ids.add(card["question_id"])
            seen_hashes.add(qhash)
            # Validate subject
            if card.get("subject") not in ("physics", "circuits", "linalg", "calculus"):
                card["subject"] = "calculus"
            fixed_cards.append(card)
            fixed_count += 1

        time.sleep(0.2)

    # Write
    OUTPUT_FILE.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in fixed_cards) + "\n", encoding="utf-8")
    print(f"\nDone: {fixed_count} cards written to solved_examples_fixed.jsonl")
    print(f"Dedup removed: {total - fixed_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
