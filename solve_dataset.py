"""Batch solve competition dataset and generate submission JSON.

Usage:
    # Solve a single dataset file
    python solve_dataset.py --input data/my_exercises.json --output results.json

    # Solve all course datasets and merge
    python solve_dataset.py \
        --input data/physics.json data/calculus.json data/linalg.json data/circuits.json \
        --output submission.json \
        --mode auto \
        --max-concurrent 5

    # Tool-only mode (fastest, no LLM API calls)
    python solve_dataset.py --input data/dev/dev.json --output results.json --mode tool_only

    # Parallel processing with controlled concurrency
    python solve_dataset.py --input data/dev/dev.json --output results.json --max-concurrent 5
    python solve_dataset.py --input 验证集/电路原理题集.json --output results.json --max-concurrent 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from eng_solver_agent.unified_agent import UnifiedAgent


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load questions from a JSON file.

    Supports both formats:
        - {"questions": [...]}
        - [...] (direct array)
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found: {candidate}")

    data = json.loads(candidate.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "questions" in data and isinstance(data["questions"], list):
            return data["questions"]
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
    raise ValueError(f"Dataset must be a list or contain a 'questions' array: {candidate}")


def save_results(path: str | Path, results: list[dict[str, Any]]) -> None:
    """Save results to a JSON file in competition format."""
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to: {candidate}")


def print_progress(current: int, total: int, start_time: float) -> None:
    """Print progress bar with ETA."""
    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / current if current > 0 else 0
    eta = avg_time * (total - current)
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "=" * filled + ">" + "." * (bar_len - filled - 1)
    print(f"\r[{bar}] {current}/{total} ({pct:.0f}%) ETA: {eta:.0f}s", end="", flush=True)


async def solve_single(
    agent: UnifiedAgent,
    question: dict[str, Any],
    mode: str,
    index: int,
    total: int,
    start_time: float,
) -> dict[str, Any]:
    """Solve a single question and return result."""
    result = agent.solve_one(question, mode=mode)
    print_progress(index + 1, total, start_time)
    return result


async def solve_dataset(
    questions: list[dict[str, Any]],
    mode: str = "auto",
    max_concurrent: int = 5,
    agent: UnifiedAgent | None = None,
) -> list[dict[str, Any]]:
    """Solve all questions with progress tracking."""
    agent = agent or UnifiedAgent()
    total = len(questions)
    start_time = time.perf_counter()

    print(f"Solving {total} questions (mode={mode}, concurrency={max_concurrent})...")

    if max_concurrent == 1:
        # Sequential solving
        results = []
        for idx, q in enumerate(questions):
            result = agent.solve_one(q, mode=mode)
            results.append(result)
            print_progress(idx + 1, total, start_time)
    else:
        # Parallel solving with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _task(q: dict[str, Any], idx: int) -> dict[str, Any]:
            async with semaphore:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, agent.solve_one, q, mode)

        tasks = [asyncio.create_task(_task(q, i)) for i, q in enumerate(questions)]
        results = []
        for idx, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            print_progress(idx + 1, total, start_time)

    elapsed = time.perf_counter() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/total:.2f}s avg)")
    return results


def run_single(
    questions: list[dict[str, Any]],
    mode: str,
    max_concurrent: int,
    agent: UnifiedAgent | None = None,
) -> list[dict[str, Any]]:
    """Run solving (sync wrapper for async)."""
    return asyncio.run(solve_dataset(questions, mode, max_concurrent, agent))


def generate_submission_info() -> dict[str, str]:
    """Generate the competition submission metadata."""
    return {
        "module": "eng_solver_agent.unified_agent",
        "class_name": "UnifiedAgent",
        "method_name": "solve",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch solve competition dataset")
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path(s) to JSON dataset file(s). Supports multiple files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "react", "legacy", "llm_only", "tool_only"],
        default="auto",
        help="Solving mode (default: auto)",
    )
    parser.add_argument(
        "--max-concurrent", "-c",
        type=int,
        default=5,
        help="Maximum concurrent LLM API calls (default: 5)",
    )
    parser.add_argument(
        "--kimi-client",
        choices=["auto", "none"],
        default="auto",
        help="LLM client configuration (default: auto)",
    )
    parser.add_argument(
        "--submission-info",
        action="store_true",
        help="Also write submission.json metadata file.",
    )

    args = parser.parse_args()

    # Load all questions from all input files
    all_questions: list[dict[str, Any]] = []
    for path in args.input:
        try:
            questions = load_questions(path)
            all_questions.extend(questions)
            print(f"Loaded {len(questions)} questions from {path}")
        except Exception as exc:
            print(f"Error loading {path}: {exc}", file=sys.stderr)
            return 1

    if not all_questions:
        print("No questions loaded.", file=sys.stderr)
        return 1

    print(f"Total questions: {len(all_questions)}\n")

    # Create agent
    agent_kwargs: dict[str, Any] = {}
    if args.kimi_client == "none":
        agent_kwargs["kimi_client"] = None

    agent = UnifiedAgent(**agent_kwargs)

    # Solve
    t0 = time.perf_counter()
    try:
        results = run_single(all_questions, args.mode, args.max_concurrent, agent)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError during solving: {exc}", file=sys.stderr)
        return 1

    # Save results
    save_results(args.output, results)

    # Optionally write submission.json
    if args.submission_info:
        submission_path = Path(args.output).parent / "submission.json"
        save_results(submission_path, [generate_submission_info()])

    # Print summary
    total_elapsed = time.perf_counter() - t0
    success_count = sum(1 for r in results if "Error" not in r.get("answer", "") and "暂无法" not in r.get("answer", ""))
    n = len(results)
    print(f"\n{'='*64}")
    print(f"  答题完成: {success_count}/{n} 成功 | 总耗时: {total_elapsed:.1f}s | 平均每道题: {total_elapsed/n:.1f}s")
    print(f"{'='*64}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
