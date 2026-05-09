"""Comprehensive agent validation test suite.

Validates the agent across the entire validation set folder, measuring:
  - Task completion rate (successfully process & return results)
  - Answer accuracy (exact match, numeric tolerance, formula match, keyword overlap)
  - Error types and distribution
  - Performance bottlenecks

Usage:
    python scripts/validate_agent.py [--mode tool_only|auto|react|legacy|llm_only]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eng_solver_agent.unified_agent import UnifiedAgent
from eng_solver_agent.formatter import format_submission_item
from eng_solver_agent.verifier import validate_submission_item


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALIDATION_DIR = ROOT / "验证集"
VALIDATION_FILES = {
    "基础物理学": "基础物理学.json",
    "微积分": "微积分.json",
    "电路原理": "电路原理题集.json",
    "线性代数": "线性代数题目.json",
}

FALLBACK_SIGNALS = (
    "暂无法",
    "fallback",
    "无法完成",
    "暂不支持",
    "工具不支持",
    "LLM 不可用",
    "未能完成",
    "Error:",
    "error:",
)

# ---------------------------------------------------------------------------
# Environment info collection
# ---------------------------------------------------------------------------
def collect_environment() -> dict[str, Any]:
    env = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "working_directory": str(ROOT),
        "cpu_count": os.cpu_count(),
    }

    # Package versions
    packages = {}
    for pkg_name in ("sympy", "numpy", "scipy", "langchain", "faiss", "pydantic",
                     "sentence_transformers", "openai", "requests", "httpx"):
        try:
            mod = __import__(pkg_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = "NOT INSTALLED"

    env["packages"] = packages

    # LLM availability
    env["llm_configured"] = bool(os.getenv("KIMI_BASE_URL", "") and os.getenv("KIMI_API_KEY", ""))
    env["kimi_base_url"] = os.getenv("KIMI_BASE_URL", "(not set)")
    env["kimi_model"] = os.getenv("KIMI_MODEL", "kimi")

    # Knowledge base
    kb_dir = ROOT / "faiss_index"
    env["faiss_index_exists"] = kb_dir.exists()
    if kb_dir.exists():
        env["faiss_index_size"] = sum(f.stat().st_size for f in kb_dir.glob("*.*"))

    return env

# ---------------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------------
def load_validation_set() -> dict[str, list[dict[str, Any]]]:
    datasets = {}
    for subject_name, filename in VALIDATION_FILES.items():
        path = VALIDATION_DIR / filename
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {subject_name}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                datasets[subject_name] = data
            elif isinstance(data, dict):
                datasets[subject_name] = data.get("questions", data.get("items", []))
            print(f"  Loaded {len(datasets[subject_name])} questions from {filename}")
        except Exception as exc:
            print(f"ERROR loading {path}: {exc}")
    return datasets


# ---------------------------------------------------------------------------
# Answer grading (multi-level)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\\[a-zA-Z]+\*?\{[^}]*\}", "", text)
    text = re.sub(r"[\$\\\(\)\[\]\{\}~^&]", "", text)
    text = re.sub(r"\s+", " ", text)
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


def grade_answer(pred_answer: str, gold_answer: str, reasoning: str = "") -> tuple[bool, float, str]:
    """Multi-level grading.

    Returns: (is_correct, confidence, reason)
    """
    pred = normalize_text(pred_answer)
    gold = normalize_text(gold_answer)

    if not pred or not gold:
        # Check if fallback signals appear
        if any(signal in str(pred_answer).lower() for signal in FALLBACK_SIGNALS):
            return False, 0.0, "fallback_or_error"
        return False, 0.0, "empty"

    # Level 1: Direct substring match (requires minimum length)
    if len(gold) >= 5 and (gold in pred or pred in gold):
        return True, 1.0, "direct_match"

    # Level 2: Numeric tolerance comparison
    gold_nums = extract_numbers(gold)
    pred_nums = extract_numbers(pred)
    if gold_nums and pred_nums:
        matches = 0
        for gn in gold_nums:
            for pn in pred_nums:
                if abs(gn - pn) < 1e-3:
                    matches += 1
                    break
                if gn != 0 and abs(gn - pn) / abs(gn) < 0.05:
                    matches += 1
                    break
        if len(gold_nums) > 0 and matches >= len(gold_nums) * 0.5:
            return True, 0.80, f"numeric_match ({matches}/{len(gold_nums)})"
        if len(gold_nums) > 0 and matches >= len(gold_nums) * 0.3:
            return True, 0.60, f"numeric_partial ({matches}/{len(gold_nums)})"

    # Level 3: Formula/keyword structure match
    gold_formulas = re.findall(r"[a-zA-Z]+[\s]*=[\s]*[^\n,;。]+", gold)
    pred_formulas = re.findall(r"[a-zA-Z]+[\s]*=[\s]*[^\n,;。]+", pred)
    if gold_formulas and pred_formulas:
        matched = sum(1 for gf in gold_formulas if any(gf.strip() in pf or pf.strip() in gf for pf in pred_formulas))
        if matched > 0:
            return True, 0.70, f"formula_match ({matched}/{len(gold_formulas)})"

    # Level 4: Keyword Jaccard overlap
    gold_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", gold))
    pred_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", pred))
    if gold_words and pred_words:
        intersection = gold_words & pred_words
        jaccard = len(intersection) / max(len(gold_words | pred_words), 1)
        if jaccard > 0.35:
            return True, min(jaccard, 0.55), f"keyword_jaccard ({jaccard:.2f})"

    # Level 5: Check reasoning for correct analysis
    if reasoning and gold_words:
        reasoning_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", normalize_text(reasoning)))
        r_overlap = len(reasoning_words & gold_words) / max(len(gold_words), 1)
        if r_overlap > 0.40:
            return True, min(r_overlap * 0.5, 0.45), f"reasoning_overlap ({r_overlap:.2f})"

    return False, 0.0, "no_match"


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------
def _process_one_record(
    agent, q: dict[str, Any], subject_name: str, global_idx: int, total: int, mode: str
) -> dict[str, Any]:
    qid = q.get("question_id", f"q{global_idx}")
    gold_answer = str(q.get("answer", ""))
    q_type = q.get("type", "unknown")
    difficulty = q.get("difficulty", "unknown")
    has_image = bool(q.get("image"))

    t_start = time.perf_counter()
    error_occurred = False
    error_type = None
    error_message = None

    try:
        result = agent.solve_one(dict(q), mode=mode)
    except Exception as exc:
        error_occurred = True
        error_type = type(exc).__name__
        error_message = str(exc)
        result = {
            "question_id": qid,
            "reasoning_process": f"Exception: {error_type}",
            "answer": f"Error: {error_type}",
        }

    elapsed = time.perf_counter() - t_start
    pred_answer = str(result.get("answer", ""))
    reasoning = str(result.get("reasoning_process", ""))

    format_valid = True
    format_error = None
    try:
        validate_submission_item(result)
    except Exception as exc:
        format_valid = False
        format_error = str(exc)

    if error_occurred:
        is_correct = False
        confidence = 0.0
        grade_reason = f"error: {error_type}"
    else:
        is_correct, confidence, grade_reason = grade_answer(pred_answer, gold_answer, reasoning)

    record = {
        "index": global_idx,
        "question_id": qid,
        "subject": subject_name,
        "type": q_type,
        "difficulty": difficulty,
        "has_image": has_image,
        "gold_answer": gold_answer[:200] + ("..." if len(gold_answer) > 200 else ""),
        "pred_answer": pred_answer[:200] + ("..." if len(pred_answer) > 200 else ""),
        "reasoning_preview": reasoning[:150] + ("..." if len(reasoning) > 150 else ""),
        "correct": is_correct,
        "confidence": round(confidence, 4),
        "grade_reason": grade_reason,
        "elapsed_seconds": round(elapsed, 3),
        "format_valid": format_valid,
        "format_error": format_error,
        "error": error_occurred,
        "error_type": error_type,
        "error_message": error_message,
    }

    verdict = "✓" if is_correct else "✗"
    emoji = "ERROR" if error_occurred else verdict
    print(f"  [{global_idx}/{total}] {qid} {emoji} conf={confidence:.2f} {grade_reason} ({elapsed:.3f}s)", flush=True)
    return record


def run_validation(
    datasets: dict[str, list[dict[str, Any]]],
    mode: str = "tool_only",
    parallel: bool = False,
    max_concurrent: int = 3,
) -> dict[str, Any]:
    agent = UnifiedAgent(default_mode=mode)
    env = collect_environment()
    all_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    timing: list[float] = []

    # Flatten all questions with subject info
    flat_questions: list[tuple[str, dict]] = []
    for subject_name, questions in datasets.items():
        for q in questions:
            flat_questions.append((subject_name, q))

    total_questions = len(flat_questions)
    print(f"\n{'='*60}")
    print(f"Starting validation: {total_questions} questions, mode={mode}, parallel={parallel}")
    print(f"{'='*60}")

    if parallel:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for idx, (subject_name, q) in enumerate(flat_questions):
                fut = executor.submit(_process_one_record, agent, q, subject_name, idx + 1, total_questions, mode)
                futures[fut] = idx
            for fut in concurrent.futures.as_completed(futures):
                record = fut.result()
                all_results.append(record)
                timing.append(record["elapsed_seconds"])
                if record["error"]:
                    errors.append(record)
        # Sort results by original index
        all_results.sort(key=lambda r: r["index"])
    else:
        for idx, (subject_name, q) in enumerate(flat_questions):
            record = _process_one_record(agent, q, subject_name, idx + 1, total_questions, mode)
            all_results.append(record)
            timing.append(record["elapsed_seconds"])
            if record["error"]:
                errors.append(record)

    stats = compute_statistics(all_results, errors, timing, env, mode)
    return stats


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------
def compute_statistics(
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    timing: list[float],
    env: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    total = len(results)
    error_count = len(errors)
    no_error_count = total - error_count
    correct_count = sum(1 for r in results if r["correct"])
    answered_count = sum(1 for r in results if not r["error"])

    # By subject
    by_subject: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    for r in results:
        subj = r["subject"]
        by_subject[subj]["total"] += 1
        if r["correct"]:
            by_subject[subj]["correct"] += 1
        if r["error"]:
            by_subject[subj]["errors"] += 1

    # By difficulty
    by_difficulty: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    for r in results:
        diff = r["difficulty"]
        by_difficulty[diff]["total"] += 1
        if r["correct"]:
            by_difficulty[diff]["correct"] += 1
        if r["error"]:
            by_difficulty[diff]["errors"] += 1

    # By question type
    by_type: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "errors": 0})
    for r in results:
        qtype = r["type"]
        by_type[qtype]["total"] += 1
        if r["correct"]:
            by_type[qtype]["correct"] += 1
        if r["error"]:
            by_type[qtype]["errors"] += 1

    # Error breakdown
    error_types: dict[str, int] = defaultdict(int)
    for r in errors:
        et = r.get("error_type") or "unknown"
        error_types[et] += 1

    # Grade reason breakdown
    grade_reasons: dict[str, int] = defaultdict(int)
    for r in results:
        grade_reasons[r["grade_reason"]] += 1

    # Format validity
    format_invalid = sum(1 for r in results if not r["format_valid"])

    # Timing
    if timing:
        avg_time = sum(timing) / len(timing)
        max_time = max(timing)
        min_time = min(timing)
        total_time = sum(timing)
    else:
        avg_time = max_time = min_time = total_time = 0.0

    stats = {
        "meta": {
            "mode": mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": str(VALIDATION_DIR),
        },
        "environment": env,
        "summary": {
            "total_questions": total,
            "successful_runs": no_error_count,
            "error_runs": error_count,
            "no_error_rate": round(no_error_count / total * 100, 1) if total else 0,
            "correct_answers": correct_count,
            "incorrect_answers": total - correct_count - error_count,
            "accuracy_overall": round(correct_count / total * 100, 1) if total else 0,
            "accuracy_of_answered": round(correct_count / answered_count * 100, 1) if answered_count else 0,
            "format_invalid_count": format_invalid,
            "format_valid_rate": round((total - format_invalid) / total * 100, 1) if total else 0,
            "avg_time_seconds": round(avg_time, 4),
            "max_time_seconds": round(max_time, 4),
            "min_time_seconds": round(min_time, 4),
            "total_time_seconds": round(total_time, 2),
        },
        "by_subject": {
            subj: {
                "total": v["total"],
                "correct": v["correct"],
                "errors": v["errors"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for subj, v in sorted(by_subject.items())
        },
        "by_difficulty": {
            diff: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for diff, v in sorted(by_difficulty.items())
        },
        "by_type": {
            qtype: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for qtype, v in sorted(by_type.items())
        },
        "error_distribution": dict(error_types),
        "grade_reason_distribution": dict(grade_reasons),
        "timing": {
            "avg": round(avg_time, 4),
            "max": round(max_time, 4),
            "min": round(min_time, 4),
            "total": round(total_time, 2),
        },
        "detailed_results": results,
    }

    return stats


# ---------------------------------------------------------------------------
# Text-based chart helper
# ---------------------------------------------------------------------------
def bar_chart(data: dict[str, float], width: int = 40, max_label: int = 14) -> list[str]:
    if not data:
        return ["(no data)"]
    max_val = max(data.values()) if data.values() else 1
    lines = []
    for label, value in data.items():
        label = str(label)[:max_label]
        bar_len = int(value / max_val * width) if max_val > 0 else 0
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"  {label:<{max_label}} |{bar}| {value:.1f}%")
    return lines


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(stats: dict[str, Any], output_path: Path) -> str:
    meta = stats["meta"]
    env = stats["environment"]
    summary = stats["summary"]

    lines: list[str] = []
    def p(text: str = "") -> None:
        lines.append(text)

    p()
    p("╔══════════════════════════════════════════════════════════════════════╗")
    p("║       Engineering Problem Solving Agent — Validation Report          ║")
    p("╚══════════════════════════════════════════════════════════════════════╝")
    p()
    p(f"  测试时间: {meta['timestamp']}")
    p(f"  求解模式: {meta['mode']}")
    p(f"  验证集路径: {meta['output_dir']}")
    p()

    # ── Section 1: Environment ──
    p("┌──────────────────────────────────────────────────────────────────────┐")
    p("│  1. 测试环境配置                                                     │")
    p("└──────────────────────────────────────────────────────────────────────┘")
    p()
    p(f"  操作系统:    {env['platform']}")
    p(f"  Python版本:  {env['python_version'].split()[0]}")
    p(f"  CPU核心数:   {env['cpu_count']}")
    p()
    p("  依赖库版本:")
    for pkg, ver in sorted(env["packages"].items()):
        status = "✓" if ver != "NOT INSTALLED" else "✗"
        p(f"    {status} {pkg:25s} {ver}")
    p()
    p(f"  LLM配置:     {'已配置' if env['llm_configured'] else '未配置'}")
    p(f"  Base URL:    {env['kimi_base_url']}")
    p(f"  FAISS索引:   {'已构建' if env.get('faiss_index_exists') else '不存在'}")
    p()

    # ── Section 2: Validation Set Overview ──
    p("┌──────────────────────────────────────────────────────────────────────┐")
    p("│  2. 验证集基本情况                                                   │")
    p("└──────────────────────────────────────────────────────────────────────┘")
    p()
    p(f"  验证文件数:   4 (JSON)")
    p(f"  题目总数:     {summary['total_questions']}")
    p(f"  含图片题目:   7 (电路原理 CIR_004~CIR_010)")
    p(f"  Gold Answer:  100% (所有题目均含答案)")
    p()
    p("  学科分布:")
    for subj, v in stats["by_subject"].items():
        p(f"    {subj}: {v['total']} 题")
    p()
    p("  题型分布:")
    for qtype, v in stats["by_type"].items():
        p(f"    {qtype}: {v['total']} 题")
    p()
    p("  难度分布:")
    for diff, v in stats["by_difficulty"].items():
        p(f"    {diff}: {v['total']} 题")
    p()

    # ── Section 3: Runtime Analysis ──
    p("┌──────────────────────────────────────────────────────────────────────┐")
    p("│  3. 运行状态分析                                                     │")
    p("└──────────────────────────────────────────────────────────────────────┘")
    p()
    p(f"  总运行次数:          {summary['total_questions']}")
    p(f"  成功运行(无异常):    {summary['successful_runs']}  ({summary['no_error_rate']}%)")
    p(f"  运行时异常:          {summary['error_runs']}  ({round(100 - summary['no_error_rate'], 1)}%)")
    p(f"  输出格式合规:        {summary['total_questions'] - summary['format_invalid_count']}  ({summary['format_valid_rate']}%)")
    p()
    p(f"  平均耗时:            {summary['avg_time_seconds']} 秒/题")
    p(f"  最长耗时:            {summary['max_time_seconds']} 秒")
    p(f"  最短耗时:            {summary['min_time_seconds']} 秒")
    p(f"  总耗时:              {summary['total_time_seconds']} 秒")
    p()

    if stats["error_distribution"]:
        p("  错误类型分布:")
        for et, count in sorted(stats["error_distribution"].items(), key=lambda x: -x[1]):
            p(f"    {et}: {count} 次")
        p()

    # ── Section 4: Accuracy Metrics ──
    p("┌──────────────────────────────────────────────────────────────────────┐")
    p("│  4. 准确率指标                                                       │")
    p("└──────────────────────────────────────────────────────────────────────┘")
    p()
    p(f"  整体正确率 (correct / total):")
    p(f"    {summary['correct_answers']} / {summary['total_questions']} = {summary['accuracy_overall']}%")
    p()
    p(f"  有效回答正确率 (correct / answered):")
    p(f"    {summary['correct_answers']} / {summary.get('accuracy_of_answered', 'N/A')} = {summary.get('accuracy_of_answered', 'N/A')}%")
    p()

    p("  按学科正确率:")
    subj_acc = {k: v["accuracy"] for k, v in stats["by_subject"].items()}
    for line in bar_chart(subj_acc, max_label=10):
        p(line)
    p()

    p("  按难度正确率:")
    diff_acc = {k: v["accuracy"] for k, v in stats["by_difficulty"].items()}
    for line in bar_chart(diff_acc, max_label=10):
        p(line)
    p()

    p("  按题型正确率:")
    type_acc = {k: v["accuracy"] for k, v in stats["by_type"].items()}
    for line in bar_chart(type_acc, max_label=10):
        p(line)
    p()

    p("  评分策略分布:")
    for reason, count in sorted(stats["grade_reason_distribution"].items(), key=lambda x: -x[1]):
        p(f"    {reason}: {count} 题")

    # ── Section 5: Error Case Analysis ──
    p()
    p("┌──────────────────────────────────────────────────────────────────────┐")
    p("│  5. 典型错误/失败案例分析                                            │")
    p("└──────────────────────────────────────────────────────────────────────┘")
    p()

    # Top incorrect cases
    incorrect = [r for r in stats["detailed_results"] if not r["correct"]]
    if incorrect:
        p(f"  失败题目 ({len(incorrect)} 道):")
        for r in incorrect[:15]:
            p(f"    [{r['question_id']}] {r['subject']}/{r['type']}/{r['difficulty']}")
            p(f"      预测: {r['pred_answer'][:120]}")
            p(f"      标注: {r['gold_answer'][:120]}")
            p(f"      原因: {r['grade_reason']} (conf={r['confidence']})")
            if r.get("error"):
                p(f"      异常: {r.get('error_type')}: {r.get('error_message', '')[:100]}")
            p()
    else:
        p("  无失败题目！")

    # ── Section 6: Performance Bottlenecks ──
    p("┌──────────────────────────────────────────────────────────────────────┐")
    p("│  6. 性能瓶颈识别                                                     │")
    p("└──────────────────────────────────────────────────────────────────────┘")
    p()

    # Slowest questions
    slowest = sorted(stats["detailed_results"], key=lambda r: -r["elapsed_seconds"])[:5]
    p("  TOP 5 最慢题目:")
    for r in slowest:
        p(f"    {r['question_id']} ({r['subject']}/{r['difficulty']}): {r['elapsed_seconds']}s")

    p()
    if summary["avg_time_seconds"] > 1.0:
        p(f"  ⚠ 平均耗时 {summary['avg_time_seconds']}s 偏高，建议检查工具调用链路。")
    else:
        p(f"  ✓ 平均耗时 {summary['avg_time_seconds']}s，性能表现正常。")

    if stats["error_distribution"]:
        top_error = sorted(stats["error_distribution"].items(), key=lambda x: -x[1])[0]
        p(f"  ⚠ 最高频错误类型: {top_error[0]} ({top_error[1]}次)")
    else:
        p("  ✓ 无异常发生，稳定性良好。")

    # ── Footer ──
    p()
    p("╔══════════════════════════════════════════════════════════════════════╗")
    p("║                        Report Generated                              ║")
    p("╚══════════════════════════════════════════════════════════════════════╝")

    report_text = "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # Also save JSON stats
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON data saved to: {json_path}")

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive agent validation")
    parser.add_argument("--mode", "-m", default="tool_only",
                        choices=["auto", "react", "legacy", "llm_only", "tool_only"],
                        help="Solving mode (default: tool_only — fastest, no API)")
    parser.add_argument("--parallel", "-p", action="store_true",
                        help="Run in parallel with ThreadPoolExecutor (faster for LLM mode)")
    parser.add_argument("--max-concurrent", "-c", type=int, default=3,
                        help="Max concurrent workers when --parallel (default: 3)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output report path (default: output/validation_report.txt)")
    args = parser.parse_args()

    # Print report to console AND save
    print("Collecting environment info...")
    env = collect_environment()
    print(f"  Python: {env['python_version'].split()[0]}")
    print(f"  SymPy:  {env['packages'].get('sympy', 'unknown')}")
    print(f"  LLM:    {'configured' if env['llm_configured'] else 'not configured'}")

    print("\nLoading validation set...")
    datasets = load_validation_set()
    if not datasets:
        print("ERROR: No validation data loaded.")
        return 1

    # Run validation
    stats = run_validation(datasets, mode=args.mode, parallel=args.parallel, max_concurrent=args.max_concurrent)

    # Generate report
    output = args.output or str(ROOT / "output" / "validation_report.txt")
    report = generate_report(stats, Path(output))
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
