"""Competition-grade solver: parallel LLM solving with smart answer grading.

Usage:
    python competition_run.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from eng_solver_agent.unified_agent import UnifiedAgent


def load_json_robust(path: str) -> list[dict[str, Any]]:
    """Load JSON with robust error handling."""
    candidate = Path(path)
    if not candidate.exists():
        return []
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
    return []


SUBJECT_FILES = {
    "基础物理学": "data/基础物理学.json",
    "线性代数": "data/线性代数题目.json",
    "微积分": "data/微积分.json",
    "电路原理": "data/电路原理题集.json",
}


def normalize_text(text: str) -> str:
    text = str(text)
    # Remove LaTeX wrappers
    text = re.sub(r"\\[a-zA-Z]+\*?\{[^}]*\}", "", text)
    text = re.sub(r"[\$\\\(\)\[\]\{\}~^]", "", text)
    # Normalize whitespace and punctuation
    text = " ".join(text.split())
    text = text.replace("，", ",").replace("。", ".").replace("；", ";")
    text = text.replace("：", ":").replace("、", ",").replace("？", "?")
    text = text.replace("！", "!").replace("·", "*")
    return text.lower().strip()


def extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"-?\d+\.?\d*", text)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def extract_formulas(text: str) -> list[str]:
    """Extract formula fragments like 'Dλ/d', '4t+3', 'v=dx/dt'."""
    # Remove Chinese text, keep formulas
    text = re.sub(r"[\u4e00-\u9fff]+", " ", text)
    # Split by punctuation
    parts = re.split(r"[。；，!\n]", text)
    formulas = []
    for part in parts:
        part = part.strip()
        # Keep parts with =, /, +, -, ^, Greek, operators
        if re.search(r"[=+\-*/^]", part) or re.search(r"[αβγδλΔπθσω∞∂∫∑∏√]", part):
            formulas.append(part)
    return formulas


def grade_answer(pred: str, gold: str) -> tuple[bool, float, str]:
    """Smart grading that handles LLM detailed responses.

    Returns: (is_correct, confidence, reason)
    """
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if not pred_norm or not gold_norm:
        return False, 0.0, "empty"

    # 1. Direct substring match (gold in pred or vice versa) - require min length
    if len(gold_norm) >= 3 and (gold_norm in pred_norm or pred_norm in gold_norm):
        return True, 1.0, "direct_match"

    # 2. Check if key formula from gold appears in pred
    gold_formulas = extract_formulas(gold)
    pred_formulas = extract_formulas(pred)
    if gold_formulas:
        matched = sum(1 for gf in gold_formulas if any(gf in pf or pf in gf for pf in pred_formulas))
        if matched > 0:
            return True, 0.85, f"formula_match ({matched}/{len(gold_formulas)})"

    # 3. Check if gold's numeric answer appears in pred
    gold_nums = extract_numbers(gold_norm)
    pred_nums = extract_numbers(pred_norm)
    if gold_nums and pred_nums:
        for gn in gold_nums:
            for pn in pred_nums:
                if abs(gn - pn) < 1e-3:
                    return True, 0.8, f"numeric_match ({gn})"
                if gn != 0 and abs(gn - pn) / abs(gn) < 0.05:
                    return True, 0.7, f"numeric_approx ({pn} vs {gn})"

    # 4. Keyword overlap for concept matching
    gold_words = set(gold_norm.split())
    pred_words = set(pred_norm.split())
    if gold_words and pred_words:
        overlap = len(gold_words & pred_words)
        union = len(gold_words | pred_words)
        jaccard = overlap / union if union > 0 else 0
        if jaccard > 0.4:
            return True, jaccard, f"keyword_jaccard ({jaccard:.2f})"

    return False, 0.0, "no_match"


async def solve_subject(
    questions: list[dict[str, Any]],
    subject_name: str,
    max_concurrent: int = 5,
) -> dict[str, Any]:
    """Solve all questions for a subject in parallel."""
    print(f"\n{'='*70}")
    print(f"Subject: {subject_name} ({len(questions)} questions)")
    print(f"{'='*70}")

    agent = UnifiedAgent()  # Real LLM from .env

    # Prepare questions (strip answers)
    solve_questions = []
    for q in questions:
        sq = dict(q)
        sq.pop("answer", None)
        solve_questions.append(sq)

    t0 = time.perf_counter()
    # Parallel solve
    results = await agent.async_solve(solve_questions, max_concurrent=max_concurrent)
    t1 = time.perf_counter()
    total_time = t1 - t0

    # Grade results
    graded = []
    correct_count = 0
    for idx, (q, r) in enumerate(zip(questions, results)):
        qid = q.get("question_id", f"q{idx}")
        gold = q.get("answer", "")
        pred = r.get("answer", "")
        q_type = q.get("type", "unknown")

        is_correct, confidence, reason = grade_answer(pred, gold)
        if is_correct:
            correct_count += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"\n[{idx+1}/{len(questions)}] {qid} ({q_type}) [{status} conf={confidence:.2f} {reason}]")
        print(f"  Pred: {pred[:100]}")
        print(f"  Gold: {gold[:100]}")

        graded.append({
            "question_id": qid,
            "type": q_type,
            "question": q.get("question", ""),
            "predicted": pred,
            "gold": gold,
            "correct": is_correct,
            "confidence": confidence,
            "reason": reason,
        })

    accuracy = correct_count / len(questions) if questions else 0.0
    avg_time = total_time / len(questions) if questions else 0.0

    print(f"\n  Summary: {correct_count}/{len(questions)} correct ({accuracy*100:.1f}%)")
    print(f"  Total time: {total_time:.1f}s | Avg per Q: {avg_time:.1f}s")

    return {
        "subject": subject_name,
        "total_questions": len(questions),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "total_time_seconds": total_time,
        "avg_time_seconds": avg_time,
        "results": graded,
    }


async def main_async() -> int:
    all_reports = []
    grand_total_q = 0
    grand_total_c = 0
    grand_total_t = 0.0

    for subject_name, path in SUBJECT_FILES.items():
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping {subject_name}")
            continue

        questions = load_json_robust(path)
        if not questions:
            print(f"Warning: No questions loaded from {path}")
            continue

        report = await solve_subject(questions, subject_name, max_concurrent=5)
        all_reports.append(report)

        grand_total_q += report["total_questions"]
        grand_total_c += report["correct_count"]
        grand_total_t += report["total_time_seconds"]

    # Grand summary
    print(f"\n{'='*70}")
    print("COMPETITION FINAL REPORT")
    print(f"{'='*70}")

    for report in all_reports:
        print(f"\n{report['subject']}:")
        print(f"  Questions: {report['total_questions']}")
        print(f"  Correct:   {report['correct_count']}/{report['total_questions']} ({report['accuracy']*100:.1f}%)")
        print(f"  Time:      {report['total_time_seconds']:.1f}s (avg {report['avg_time_seconds']:.1f}s/q)")

    acc = grand_total_c / grand_total_q if grand_total_q else 0.0
    print(f"\n{'='*70}")
    print(f"GRAND TOTAL:")
    print(f"  Total Questions: {grand_total_q}")
    print(f"  Total Correct:   {grand_total_c}/{grand_total_q} ({acc*100:.1f}%)")
    print(f"  Total Time:      {grand_total_t:.1f}s")
    print(f"  Avg Time/Q:      {grand_total_t/grand_total_q:.1f}s" if grand_total_q else "")
    print(f"{'='*70}")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grand_total": {
            "questions": grand_total_q,
            "correct": grand_total_c,
            "accuracy": acc,
            "total_time_seconds": grand_total_t,
        },
        "subjects": all_reports,
    }
    Path("output").mkdir(exist_ok=True)
    Path("output/competition_report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to: output/competition_report.json")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
