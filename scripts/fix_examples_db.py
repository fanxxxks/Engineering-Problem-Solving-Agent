"""Deduplicate and fix categorization in solved_examples.jsonl.

This script:
1. Scans for duplicate question_id values and deduplicates
2. Fixes incorrect subject categorizations (e.g. calculus tagged as physics)
3. Assigns clean, sequential numbering per subject
4. Backs up original before modifications

Usage:
    python scripts/fix_examples_db.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "eng_solver_agent" / "retrieval" / "solved_examples.jsonl"

# Known duplicate/overlap patterns
DUPLICATE_PAIRS = {
    "MATH_CALC_SERIES_001": "MATH_SERIES_001",
    "MATH_CALC_SERIES_002": "MATH_SERIES_002",
    "MATH_CALC_SERIES_003": "MATH_SERIES_003",
    "MATH_CALC_SERIES_004": "MATH_SERIES_004",
    "MATH_CALC_SERIES_005": "MATH_SERIES_005",
    "MATH_CALC_SERIES_006": "MATH_SERIES_006",
    "MATH_CALC_SERIES_007": "MATH_SERIES_007",
    "MATH_CALC_SERIES_008": "MATH_SERIES_008",
    "MATH_CALC_SERIES_009": "MATH_SERIES_009",
    "MATH_CALC_SERIES_010": "MATH_SERIES_010",
    "MATH_CALC_SERIES_011": "MATH_SERIES_011",
    "MATH_CALC_SERIES_012": "MATH_SERIES_012",
    "MATH_CALC_SERIES_013": "MATH_SERIES_013",
    "MATH_CALC_SERIES_014": "MATH_SERIES_014",
    "MATH_CALC_SERIES_015": "MATH_SERIES_015",
    "MATH_CALC_SERIES_016": "MATH_SERIES_016",
    "MATH_CALC_SERIES_017": "MATH_SERIES_017",
    "MATH_CALC_SERIES_018": "MATH_SERIES_018",
    "MATH_CALC_SERIES_019": "MATH_SERIES_019",
    "MATH_CALC_SERIES_020": "MATH_SERIES_020",
    "MATH_CALC_SERIES_021": "MATH_SERIES_021",
}

SUBJECT_KEYWORDS = {
    "physics": ["力", "速度", "加速度", "质量", "能量", "动量", "牛顿", "force", "velocity", "acceleration", "mass", "momentum", "波", "光", "电", "磁", "热", "量子", "相对论", "wave", "optics", "thermal", "quantum", "relativity", "波长", "干涉", "衍射", "光电", "德布罗意", "电动势", "磁感应", "电磁", "感应", "角动量", "转动"],
    "circuits": ["电路", "电阻", "电压", "电流", "串联", "并联", "欧姆", "circuit", "resistor", "voltage", "current", "ohm", "KCL", "KVL", "节点", "网孔", "戴维南", "诺顿", "谐振", "运放", "op-amp", "三相", "暂态", "相量", "功率因数", "换路"],
    "linalg": ["矩阵", "行列式", "特征值", "特征向量", "秩", "逆", "线性", "方程组", "消元", "matrix", "determinant", "eigen", "rank", "inverse", "linear system", "gauss", "向量", "基", "维数", "子空间", "Cayley", "韦达"],
    "calculus": ["导数", "积分", "极限", "微分", "泰勒", "级数", "derivative", "integral", "limit", "series", "taylor", "收敛", "一致收敛", "求导", "求积分", "求极限", "不定积分", "定积分", "收敛域", "收敛半径"],
}


def _safe_str(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return str(v)
    return ""

def detect_subject(question: str, reasoning: str, answer, current_subject: str) -> str:
    """Detect the likely subject from question text using keyword voting."""
    a = _safe_str(answer)
    text = f"{_safe_str(question)} {_safe_str(reasoning)[:200]} {a[:100]}".lower()
    scores = {}
    for subj, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        scores[subj] = score

    best = max(scores, key=scores.get)
    if scores[best] >= 3 and scores[best] >= scores.get(current_subject, 0) + 2:
        return best
    return current_subject


def _read_all_entries(path: Path) -> list[tuple[int, dict]]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append((line_num, obj))
    return entries


def main() -> int:
    print("=" * 60)
    print("Fixing solved_examples.jsonl: dedup + categorization")
    print("=" * 60)

    # Backup
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = EXAMPLES_PATH.with_suffix(f".backup_dedup_{ts}.jsonl")
    backup.write_text(EXAMPLES_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  Backup: {backup.name}")

    entries = _read_all_entries(EXAMPLES_PATH)
    print(f"  Loaded {len(entries)} entries")

    # Phase 1: Detect duplicates
    seen_ids: dict[str, int] = {}
    to_remove: set[int] = set()  # line numbers to remove

    for line_num, obj in entries:
        qid = obj.get("question_id", "")
        if qid in seen_ids:
            to_remove.add(line_num)
            print(f"  [DUP] {qid} at line {line_num} (first at line {seen_ids[qid]})")
        else:
            seen_ids[qid] = line_num

    # Also detect MATH_SERIES_* duplicates of MATH_CALC_SERIES_*
    for calc_id, series_id in DUPLICATE_PAIRS.items():
        if calc_id in seen_ids and series_id in seen_ids:
            to_remove.add(seen_ids[series_id])
            print(f"  [DUP-SERIES] Removing {series_id} (duplicate of {calc_id})")

    # Phase 2: Fix subject categorizations
    fixed_subjects = 0
    for line_num, obj in entries:
        if line_num in to_remove:
            continue
        old_subject = obj.get("subject", "")
        new_subject = detect_subject(
            obj.get("question", ""),
            obj.get("reasoning_process", ""),
            obj.get("answer", ""),
            old_subject,
        )
        if new_subject != old_subject:
            obj["subject"] = new_subject
            fixed_subjects += 1
            print(f"  [FIX] Line {line_num} {obj['question_id']}: {old_subject} → {new_subject}")

    # Phase 3: Re-assign clean sequential IDs
    subject_counters: dict[str, int] = {}
    for line_num, obj in entries:
        if line_num in to_remove:
            continue
        subj = obj["subject"]
        subject_counters[subj] = subject_counters.get(subj, 0) + 1
        old_id = obj["question_id"]
        new_id = f"{subj}-ex-{subject_counters[subj]:03d}"
        obj["question_id"] = new_id
        if old_id != new_id:
            print(f"  [RENAME] Line {line_num}: {old_id} → {new_id}")

    # Phase 4: Write cleaned file
    cleaned = [obj for line_num, obj in entries if line_num not in to_remove]
    with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
        for obj in cleaned:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Phase 5: Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Original entries: {len(entries)}")
    print(f"  Duplicates removed: {len(to_remove)}")
    print(f"  Subject fixes: {fixed_subjects}")
    print(f"  Final entries: {len(cleaned)}")
    for subj, count in sorted(subject_counters.items()):
        print(f"    {subj}: {count}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
