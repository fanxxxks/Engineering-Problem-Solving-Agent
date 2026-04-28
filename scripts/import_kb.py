"""Import data/al.json, ph.json, calculus.json into the knowledge base.

This script reads the three competition dataset files, converts each problem
into the solved_examples.jsonl format, and appends them to the existing
knowledge base. Then rebuilds the FAISS index.

The input JSON files may have structural issues:
1. LaTeX backslashes that overlap with JSON escape sequences
2. Missing commas between array elements
3. Extra data after the closing bracket

Usage:
    python scripts/import_kb.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILE_MAP = {
    "calculus.json": "calculus",
    "al.json": "linalg",
    "ph.json": "physics",
}

TOPIC_PATTERNS: dict[str, list[tuple[str, list[str]]]] = {
    "calculus": [
        ("series_convergence", ["级数", "收敛域", "收敛半径", "series", "convergence"]),
        ("uniform_convergence", ["一致收敛", "uniform convergence"]),
        ("derivative", ["导数", "求导", "derivative", "微分", "\\varphi'"]),
        ("integral", ["积分", "integral", "\\int", "定积分", "重积分"]),
        ("limit", ["极限", "limit"]),
        ("taylor", ["泰勒", "taylor", "展开"]),
        ("continuity", ["连续", "连续性", "continuous"]),
        ("proof", ["证明", "求证", "prove"]),
    ],
    "linalg": [
        ("determinant", ["行列式", "determinant", "det"]),
        ("linear_system", ["方程组", "线性方程组", "消元", "Gauss", "无解", "唯一解", "无穷多解"]),
        ("matrix_operations", ["矩阵", "方阵", "matrix", "幂", "求逆"]),
        ("eigenvalues", ["特征值", "特征向量", "eigen"]),
        ("rank", ["秩", "rank"]),
        ("inverse", ["逆", "inverse"]),
        ("linear_independence", ["线性相关", "线性无关", "线性组合"]),
        ("vector_space", ["向量", "空间", "子空间", "基", "维数"]),
        ("quadratic_form", ["二次型", "正定", "负定"]),
    ],
    "physics": [
        ("kinematics", ["运动", "速度", "加速度", "位移", "路程", "上抛", "平抛", "匀速", "匀变速"]),
        ("newton_laws", ["牛顿", "newton", "受力", "合力"]),
        ("momentum", ["动量", "冲量", "momentum", "impulse"]),
        ("energy", ["能量", "动能", "势能", "功", "energy", "work"]),
        ("vectors", ["向量", "矢量", "叉乘", "点乘", "混合积", "\\vec", "\\hat"]),
        ("circular_motion", ["圆周", "曲线运动", "法向", "切向", "向心"]),
        ("proof", ["证明", "求证", "推导"]),
        ("oscillation", ["振动", "弹簧", "周期", "oscillation"]),
        ("rotation", ["转动", "圆盘", "角速度", "角动量", "rotation"]),
    ],
}


def load_json_robust(filepath: Path) -> list[dict]:
    """Load a JSON file that may have LaTeX + structural issues.

    Strategy: extract each top-level object by finding balanced {...} pairs,
    then parse each one individually after fixing LaTeX backslashes.
    """
    raw = filepath.read_text(encoding="utf-8")

    # Step 0: Fix raw newlines inside JSON strings (before any other processing).
    chars = []
    i = 0
    in_string = False
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            in_string = not in_string
            chars.append(ch)
        elif ch == '\n' and in_string:
            chars.append('\ue000')
        else:
            chars.append(ch)
        i += 1
    raw = ''.join(chars)

    # Step 1: Extract balanced top-level JSON objects from the array.
    # We look for each '{' ... '}' pair at bracket_depth==1.
    objects = []
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape_next = False
    obj_start = -1
    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
        elif ch == '{':
            if bracket_depth == 1 and brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if bracket_depth == 1 and brace_depth == 0 and obj_start >= 0:
                objects.append(raw[obj_start:i + 1])
                obj_start = -1

    if not objects:
        raise ValueError(f"No JSON objects found in {filepath.name}")

    # Step 2: Fix LaTeX backslashes in each object and parse
    results = []
    for obj_text in objects:
        # Fix the backslashes
        obj_text = obj_text.replace('\\', '\\\\')
        # Restore newline sentinel
        obj_text = obj_text.replace('\ue000', '\\n')
        # Fix missing comma before closing brace if present
        obj_text = re.sub(r'"\s*\n\s*\}', r'"\n    }', obj_text)
        try:
            obj = json.loads(obj_text)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue

    return results


def infer_topic(subject: str, question_text: str) -> str:
    text = question_text.lower()
    for topic, keywords in TOPIC_PATTERNS.get(subject, []):
        if any(kw in text for kw in keywords):
            return topic
    return "general"


def extract_tags(subject: str, question_text: str) -> list[str]:
    tags = [subject]
    text = question_text.lower()
    for topic, keywords in TOPIC_PATTERNS.get(subject, []):
        if any(kw in text for kw in keywords):
            if topic not in tags:
                tags.append(topic)
    return tags


def extract_final_answer(answer_text: str) -> str:
    """Try to extract the final concise answer from detailed answer text."""
    if not answer_text:
        return ""
    text = answer_text.strip()

    patterns = [
        r"最终结果[：:]\s*(.+?)(?:\n|$)",
        r"最终答案[：:]\s*(.+?)(?:\n|$)",
        r"最终收敛域[：:]\s*(.+?)(?:\n|$)",
        r"因此最终结果[：:]\s*(.+?)(?:\n|$)",
        r"正确选项为\*{0,2}([A-D]+)\*{0,2}",
        r"正确选项[是为：:]\s*([A-D]+)",
        r"### 最终\w*\s*\n(.+?)(?:\n|$)",
        r"综上[，,]\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            short = match.group(1).strip()
            if 3 < len(short) < 500:
                return short

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return lines[-1]

    return text


def convert_entry(entry: dict, subject: str) -> dict:
    question_text = entry.get("question", "")
    answer_text = entry.get("answer", "")
    topic = infer_topic(subject, question_text)
    tags = extract_tags(subject, question_text)

    reasoning = answer_text
    short_answer = extract_final_answer(answer_text)
    if len(short_answer) > 500 or len(short_answer) < 3:
        short_answer = answer_text[:500]

    return {
        "question_id": entry.get("question_id", ""),
        "subject": subject,
        "topic": topic,
        "question": question_text.strip(),
        "reasoning_process": reasoning.strip(),
        "answer": short_answer.strip(),
        "tags": tags,
    }


def main():
    kb_dir = ROOT / "eng_solver_agent" / "retrieval"
    examples_path = kb_dir / "solved_examples.jsonl"

    # Load existing examples
    existing = []
    if examples_path.exists():
        with open(examples_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    existing_ids = {ex["question_id"] for ex in existing}
    print(f"现有知识库条目: {len(existing)}")

    # Process each data file
    total_added = 0
    skipped = 0

    for filename, subject in FILE_MAP.items():
        filepath = ROOT / "data" / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename} 不存在")
            continue

        print(f"  正在加载 {filename} ...")
        data = load_json_robust(filepath)
        print(f"  {filename}: {len(data)} 条记录已加载")

        added = 0
        file_skipped = 0
        for entry in data:
            qid = entry.get("question_id", "")
            if qid in existing_ids:
                file_skipped += 1
                continue

            converted = convert_entry(entry, subject)
            existing.append(converted)
            existing_ids.add(qid)
            added += 1

        total_added += added
        skipped += file_skipped
        print(f"    → 新增 {added} 条, 跳过 {file_skipped} 条 (重复)")

    # Write updated knowledge base
    with open(examples_path, "w", encoding="utf-8") as f:
        for ex in existing:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n完成! 总计: {len(existing)} 条知识库条目 ({total_added} 条新增, {skipped} 条重复)")
    print(f"知识库保存至: {examples_path}")

    # Print topic distribution
    topics = Counter(ex["topic"] for ex in existing)
    subjects = Counter(ex["subject"] for ex in existing)
    print(f"\n学科分布: {dict(subjects)}")
    print(f"主题分布: {dict(topics)}")


if __name__ == "__main__":
    main()
