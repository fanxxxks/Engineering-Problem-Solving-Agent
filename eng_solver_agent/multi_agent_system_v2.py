"""Multi-Agent Parallel Processing System V2 - Simplified and Efficient.

Architecture:
- Orchestrator: Manages parallel execution
- Worker: Simple LLM-based solver
- Checker: Validates results
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from typing import Any

from eng_solver_agent.llm.kimi_client import KimiClient
from eng_solver_agent.formatter import format_submission_item


@dataclass
class WorkerResult:
    """Result from a worker agent."""
    question_id: str
    success: bool
    answer: str = ""
    reasoning: str = ""
    elapsed_time: float = 0.0
    error_message: str = ""
    confidence: float = 0.0


@dataclass
class CheckResult:
    """Result from checker agent."""
    question_id: str
    is_valid: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    suggestions: str = ""


class SimpleWorker:
    """Simplified worker that uses LLM directly."""
    
    def __init__(self, worker_id: int, kimi_client: KimiClient):
        self.worker_id = worker_id
        self.kimi_client = kimi_client
    
    def solve(self, question: dict[str, Any]) -> WorkerResult:
        """Solve a single question using LLM."""
        question_id = question.get("question_id", f"unknown_{self.worker_id}")
        question_text = question.get("question", "")
        
        start_time = time.time()
        
        try:
            # Direct LLM solving
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert engineering problem solver.

Instructions:
1. Read the problem carefully
2. Show step-by-step reasoning
3. Provide the final answer
4. Return response in strict JSON format

Response format:
{
    "reasoning_process": "detailed step-by-step solution",
    "answer": "final numerical answer or expression"
}"""
                },
                {
                    "role": "user",
                    "content": f"Solve this problem:\n\n{question_text}"
                }
            ]
            
            response = self.kimi_client.chat_json(
                messages,
                required_keys=["reasoning_process", "answer"]
            )
            
            elapsed = time.time() - start_time
            
            reasoning = response.get("reasoning_process", "").strip()
            answer = response.get("answer", "").strip()
            
            confidence = self._estimate_confidence(answer, reasoning)
            
            return WorkerResult(
                question_id=question_id,
                success=True,
                answer=answer,
                reasoning=reasoning,
                elapsed_time=elapsed,
                confidence=confidence
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return WorkerResult(
                question_id=question_id,
                success=False,
                elapsed_time=elapsed,
                error_message=str(e)
            )
    
    def _estimate_confidence(self, answer: str, reasoning: str) -> float:
        """Estimate confidence based on answer quality."""
        confidence = 0.5
        
        # Higher confidence if answer contains numbers
        if any(c.isdigit() for c in answer):
            confidence += 0.2
        
        # Higher confidence if reasoning is detailed
        if len(reasoning) > 100:
            confidence += 0.15
        
        # Lower confidence if contains uncertainty markers
        uncertainty_markers = ["无法", "unknown", "error", "失败", "暂无法"]
        if any(marker in answer.lower() for marker in uncertainty_markers):
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))


class SimpleChecker:
    """Simplified checker using rule-based validation."""
    
    def check_answer(self, question: dict[str, Any], worker_result: WorkerResult) -> CheckResult:
        """Check if an answer is valid."""
        question_id = worker_result.question_id
        
        if not worker_result.success:
            return CheckResult(
                question_id=question_id,
                is_valid=False,
                confidence=0.0,
                issues=["Worker failed"],
                suggestions="Retry"
            )
        
        issues = []
        answer = worker_result.answer
        
        # Check for empty answer
        if not answer or len(answer.strip()) < 3:
            issues.append("Answer too short")
        
        # Check for error messages
        error_markers = ["error", "失败", "无法", "unknown", "exception", "暂无法"]
        if any(marker in answer.lower() for marker in error_markers):
            issues.append("Contains error indicators")
        
        is_valid = len(issues) == 0 and worker_result.confidence > 0.5
        
        return CheckResult(
            question_id=question_id,
            is_valid=is_valid,
            confidence=worker_result.confidence,
            issues=issues,
            suggestions="Review manually" if issues else ""
        )


class MultiAgentOrchestratorV2:
    """Simplified orchestrator with true parallel processing."""
    
    def __init__(
        self,
        max_workers: int = 3,
        kimi_client: KimiClient | None = None,
        enable_checker: bool = True
    ):
        self.max_workers = max_workers
        self.kimi_client = kimi_client or KimiClient()
        self.enable_checker = enable_checker
        self.checker = SimpleChecker() if enable_checker else None
    
    def solve_batch(
        self,
        questions: list[dict[str, Any]],
        max_retries: int = 0
    ) -> list[dict[str, Any]]:
        """Solve multiple questions in parallel."""
        print(f"\n{'='*80}")
        print(f"Multi-Agent Parallel Processing V2")
        print(f"{'='*80}")
        print(f"Total questions: {len(questions)}")
        print(f"Max workers: {self.max_workers}")
        print(f"Checker enabled: {self.enable_checker}\n")
        
        # Phase 1: Parallel solving
        print("Phase 1: Parallel solving...")
        start_time = time.time()
        worker_results = self._parallel_solve(questions)
        solve_time = time.time() - start_time
        
        print(f"\nSolving completed in {solve_time:.1f}s")
        
        # Phase 2: Check answers
        if self.enable_checker and self.checker:
            print("\nPhase 2: Checking answers...")
            check_results = self._check_answers(questions, worker_results)
            
            # Phase 3: Retry if needed
            if max_retries > 0:
                retry_count = self._retry_if_needed(questions, worker_results, check_results)
                if retry_count > 0:
                    print(f"\nRetried {retry_count} questions")
        
        # Convert to final format
        final_results = self._convert_to_final_format(worker_results)
        
        # Print summary
        self._print_summary(worker_results, solve_time)
        
        return final_results
    
    def _parallel_solve(self, questions: list[dict[str, Any]]) -> list[WorkerResult]:
        """Solve questions in parallel using thread pool."""
        results = [None] * len(questions)  # Pre-allocate to maintain order
        
        def solve_single(idx: int, question: dict) -> tuple[int, WorkerResult]:
            """Solve a single question and return with index."""
            worker = SimpleWorker(idx, self.kimi_client)
            result = worker.solve(question)
            return idx, result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(solve_single, i, q): i 
                for i, q in enumerate(questions)
            }
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                completed += 1
                
                try:
                    _, result = future.result()
                    results[idx] = result
                    
                    status = "✓" if result.success else "✗"
                    print(f"  [{completed}/{len(questions)}] {status} {result.question_id} "
                          f"({result.elapsed_time:.1f}s, conf: {result.confidence:.2f})")
                except Exception as e:
                    question_id = questions[idx].get("question_id", f"q_{idx}")
                    results[idx] = WorkerResult(
                        question_id=question_id,
                        success=False,
                        error_message=str(e)
                    )
                    print(f"  [{completed}/{len(questions)}] ✗ {question_id} (Exception)")
        
        return results
    
    def _check_answers(
        self,
        questions: list[dict[str, Any]],
        worker_results: list[WorkerResult]
    ) -> list[CheckResult]:
        """Check all answers."""
        check_results = []
        
        for question, result in zip(questions, worker_results):
            check_result = self.checker.check_answer(question, result)
            check_results.append(check_result)
            
            status = "✓" if check_result.is_valid else "?"
            print(f"  {status} {result.question_id}: valid={check_result.is_valid}")
        
        return check_results
    
    def _retry_if_needed(
        self,
        questions: list[dict[str, Any]],
        worker_results: list[WorkerResult],
        check_results: list[CheckResult]
    ) -> int:
        """Retry failed or low-confidence answers."""
        retry_count = 0
        
        for i, (check_result, result) in enumerate(zip(check_results, worker_results)):
            if not check_result.is_valid or result.confidence < 0.5:
                print(f"  Retrying {result.question_id}...")
                
                # Create new worker and retry
                worker = SimpleWorker(999, self.kimi_client)
                new_result = worker.solve(questions[i])
                
                if new_result.success and new_result.confidence > result.confidence:
                    worker_results[i] = new_result
                    retry_count += 1
                    print(f"    → Improved to {new_result.confidence:.2f}")
        
        return retry_count
    
    def _convert_to_final_format(self, worker_results: list[WorkerResult]) -> list[dict[str, Any]]:
        """Convert worker results to final submission format."""
        final_results = []
        
        for result in worker_results:
            if result.success:
                final_results.append(format_submission_item(
                    question_id=result.question_id,
                    reasoning_process=result.reasoning,
                    answer=result.answer
                ))
            else:
                final_results.append(format_submission_item(
                    question_id=result.question_id,
                    reasoning_process=f"Failed: {result.error_message}",
                    answer=f"Error: {result.error_message}"
                ))
        
        return final_results
    
    def _print_summary(self, worker_results: list[WorkerResult], total_time: float):
        """Print processing summary."""
        total = len(worker_results)
        successful = sum(1 for r in worker_results if r.success)
        avg_confidence = sum(r.confidence for r in worker_results) / total if total > 0 else 0
        
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Total questions: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {total - successful}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Throughput: {total/total_time:.2f} questions/sec")
        print(f"{'='*80}\n")


# Convenience function
def solve_batch_parallel_v2(
    questions: list[dict[str, Any]],
    max_workers: int = 3,
    enable_checker: bool = True
) -> list[dict[str, Any]]:
    """Solve multiple questions in parallel."""
    orchestrator = MultiAgentOrchestratorV2(
        max_workers=max_workers,
        enable_checker=enable_checker
    )
    return orchestrator.solve_batch(questions)
