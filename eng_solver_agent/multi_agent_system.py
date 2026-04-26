"""Multi-Agent Parallel Processing System for Engineering Problem Solving.

Architecture:
- Orchestrator: Distributes questions to workers and aggregates results
- Worker Agent: Solves individual questions using LLM + tools
- Checker Agent: Validates answers and identifies suspicious results
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from typing import Any

from eng_solver_agent.agent_v2 import EnhancedSolverAgent
from eng_solver_agent.debug_logger import log_pipeline_stage, section, step
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


class WorkerAgent:
    """Worker agent that solves a single question."""
    
    def __init__(self, worker_id: int, kimi_client: KimiClient | None = None):
        self.worker_id = worker_id
        self.solver = EnhancedSolverAgent(kimi_client=kimi_client)
    
    def solve(self, question: dict[str, Any]) -> WorkerResult:
        """Solve a single question."""
        question_id = question.get("question_id", f"unknown_{self.worker_id}")
        start_time = time.time()
        step("WorkerAgent", f"Worker#{self.worker_id} 开始解题: {question_id}", color="cyan")

        try:
            # Try LLM direct solve first
            result = self.solver.solve_one(question)
            elapsed = time.time() - start_time

            step("WorkerAgent", f"Worker#{self.worker_id} 完成: {question_id} ({elapsed:.1f}s)", color="green")
            return WorkerResult(
                question_id=question_id,
                success=True,
                answer=result.get("answer", ""),
                reasoning=result.get("reasoning_process", ""),
                elapsed_time=elapsed,
                confidence=self._estimate_confidence(result)
            )

        except Exception as e:
            elapsed = time.time() - start_time
            step("WorkerAgent", f"Worker#{self.worker_id} 失败: {question_id} - {e}", color="red")
            return WorkerResult(
                question_id=question_id,
                success=False,
                elapsed_time=elapsed,
                error_message=str(e)
            )
    
    def _estimate_confidence(self, result: dict) -> float:
        """Estimate confidence based on answer quality."""
        answer = result.get("answer", "")
        reasoning = result.get("reasoning_process", "")
        
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


class CheckerAgent:
    """Checker agent that validates answers."""
    
    def __init__(self, kimi_client: KimiClient | None = None):
        self.kimi_client = kimi_client or KimiClient()
    
    def check_answer(self, question: dict[str, Any], worker_result: WorkerResult) -> CheckResult:
        """Check if an answer is valid and reasonable."""
        question_id = worker_result.question_id
        
        # If worker failed, no need to check
        if not worker_result.success:
            return CheckResult(
                question_id=question_id,
                is_valid=False,
                confidence=0.0,
                issues=["Worker failed to produce answer"],
                suggestions="Retry with different approach"
            )
        
        # Use LLM to check answer quality
        try:
            return self._llm_check(question, worker_result)
        except Exception:
            # Fallback to rule-based check
            return self._rule_based_check(question, worker_result)
    
    def _llm_check(self, question: dict[str, Any], worker_result: WorkerResult) -> CheckResult:
        """Use LLM to check answer quality."""
        step("CheckerAgent", f"[检查] 检查答案: {worker_result.question_id}", color="magenta")
        question_text = question.get("question", "")
        answer = worker_result.answer
        reasoning = worker_result.reasoning

        messages = [
            {
                "role": "system",
                "content": """You are an expert answer checker. Evaluate if the solution is correct and reasonable.

Check for:
1. Mathematical correctness
2. Unit consistency
3. Reasonable magnitude
4. Complete reasoning

Respond in strict JSON:
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of issues if any"],
    "suggestions": "improvement suggestions"
}"""
            },
            {
                "role": "user",
                "content": f"Question: {question_text[:500]}\n\nAnswer: {answer[:500]}\n\nReasoning: {reasoning[:1000]}"
            }
        ]
        
        response = self.kimi_client.chat_json(
            messages,
            required_keys=["is_valid", "confidence"]
        )

        step("CheckerAgent", f"[成功] 检查结果: valid={response.get('is_valid')}, confidence={response.get('confidence'):.2f}", color="green")
        return CheckResult(
            question_id=worker_result.question_id,
            is_valid=response.get("is_valid", False),
            confidence=float(response.get("confidence", 0.0)),
            issues=response.get("issues", []),
            suggestions=response.get("suggestions", "")
        )
    
    def _rule_based_check(self, question: dict[str, Any], worker_result: WorkerResult) -> CheckResult:
        """Rule-based check as fallback."""
        issues = []
        
        answer = worker_result.answer
        
        # Check for empty answer
        if not answer or len(answer.strip()) < 3:
            issues.append("Answer is too short or empty")
        
        # Check for error messages
        error_markers = ["error", "失败", "无法", "unknown", "exception"]
        if any(marker in answer.lower() for marker in error_markers):
            issues.append("Answer contains error indicators")
        
        # Check for reasonable length
        if len(answer) > 1000:
            issues.append("Answer is suspiciously long")
        
        is_valid = len(issues) == 0 and worker_result.confidence > 0.5
        
        return CheckResult(
            question_id=worker_result.question_id,
            is_valid=is_valid,
            confidence=worker_result.confidence,
            issues=issues,
            suggestions="Review answer manually" if issues else ""
        )


class MultiAgentOrchestrator:
    """Orchestrator that manages multiple worker agents."""
    
    def __init__(
        self,
        max_workers: int = 4,
        kimi_client: KimiClient | None = None,
        enable_checker: bool = True
    ):
        self.max_workers = max_workers
        self.kimi_client = kimi_client or KimiClient()
        self.enable_checker = enable_checker
        self.checker = CheckerAgent(self.kimi_client) if enable_checker else None
    
    def solve_batch(
        self,
        questions: list[dict[str, Any]],
        max_retries: int = 1
    ) -> list[dict[str, Any]]:
        """Solve multiple questions using parallel workers."""
        print(f"\n{'='*80}")
        print(f"Multi-Agent Parallel Processing")
        print(f"{'='*80}")
        print(f"Total questions: {len(questions)}")
        print(f"Max workers: {self.max_workers}")
        print(f"Checker enabled: {self.enable_checker}\n")
        
        # Phase 1: Parallel solving
        print("Phase 1: Parallel solving...")
        worker_results = self._parallel_solve(questions)
        
        # Phase 2: Check answers (if enabled)
        if self.enable_checker and self.checker:
            print("\nPhase 2: Checking answers...")
            check_results = self._check_answers(questions, worker_results)
            
            # Phase 3: Retry failed/suspicious answers
            if max_retries > 0:
                print("\nPhase 3: Retrying suspicious answers...")
                worker_results = self._retry_failed(
                    questions, worker_results, check_results, max_retries
                )
        
        # Convert to final format
        final_results = self._convert_to_final_format(worker_results)
        
        # Print summary
        self._print_summary(worker_results)
        
        return final_results
    
    def _parallel_solve(self, questions: list[dict[str, Any]]) -> list[WorkerResult]:
        """Solve questions in parallel using thread pool."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_question = {}
            for i, question in enumerate(questions):
                worker = WorkerAgent(i, self.kimi_client)
                future = executor.submit(worker.solve, question)
                future_to_question[future] = (i, question)
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_question):
                idx, question = future_to_question[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    status = "✓" if result.success else "✗"
                    print(f"  [{completed}/{len(questions)}] {status} {result.question_id} "
                          f"({result.elapsed_time:.1f}s, conf: {result.confidence:.2f})")
                except Exception as e:
                    question_id = question.get("question_id", f"q_{idx}")
                    results.append(WorkerResult(
                        question_id=question_id,
                        success=False,
                        error_message=str(e)
                    ))
                    print(f"  [{completed}/{len(questions)}] ✗ {question_id} (Exception: {e})")
        
        # Sort results by question_id to maintain order
        results.sort(key=lambda r: r.question_id)
        return results
    
    def _check_answers(
        self,
        questions: list[dict[str, Any]],
        worker_results: list[WorkerResult]
    ) -> list[CheckResult]:
        """Check all answers."""
        check_results = []
        
        # Create question lookup
        question_map = {q.get("question_id", f"q_{i}"): q 
                       for i, q in enumerate(questions)}
        
        for result in worker_results:
            question = question_map.get(result.question_id, {})
            check_result = self.checker.check_answer(question, result)
            check_results.append(check_result)
            
            status = "✓" if check_result.is_valid else "?"
            print(f"  {status} {result.question_id}: "
                  f"valid={check_result.is_valid}, "
                  f"conf={check_result.confidence:.2f}")
            
            if check_result.issues:
                for issue in check_result.issues:
                    print(f"      Issue: {issue}")
        
        return check_results
    
    def _retry_failed(
        self,
        questions: list[dict[str, Any]],
        worker_results: list[WorkerResult],
        check_results: list[CheckResult],
        max_retries: int
    ) -> list[WorkerResult]:
        """Retry failed or suspicious answers."""
        question_map = {q.get("question_id", f"q_{i}"): q
                       for i, q in enumerate(questions)}

        results_map = {r.question_id: r for r in worker_results}

        for attempt in range(max_retries):
            improved_any = False
            for check_result in check_results:
                if not check_result.is_valid or check_result.confidence < 0.6:
                    question_id = check_result.question_id
                    question = question_map.get(question_id)

                    if question is None:
                        continue

                    print(f"  Retrying {question_id} (attempt {attempt + 1}/{max_retries})...")

                    worker = WorkerAgent(999, self.kimi_client)
                    new_result = worker.solve(question)

                    if new_result.success and new_result.confidence > results_map[question_id].confidence:
                        results_map[question_id] = new_result
                        improved_any = True
                        print(f"    → Improved: {new_result.confidence:.2f}")
                    else:
                        print(f"    → No improvement")
            if not improved_any:
                break

        return list(results_map.values())
    
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
                # Return error response
                final_results.append(format_submission_item(
                    question_id=result.question_id,
                    reasoning_process=f"Failed to solve: {result.error_message}",
                    answer=f"Error: {result.error_message}"
                ))
        
        return final_results
    
    def _print_summary(self, worker_results: list[WorkerResult]):
        """Print processing summary."""
        total = len(worker_results)
        successful = sum(1 for r in worker_results if r.success)
        total_time = sum(r.elapsed_time for r in worker_results)
        avg_confidence = sum(r.confidence for r in worker_results) / total if total > 0 else 0
        
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Total questions: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {total - successful}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"{'='*80}\n")


# Convenience function
def solve_batch_parallel(
    questions: list[dict[str, Any]],
    max_workers: int = 4,
    enable_checker: bool = True
) -> list[dict[str, Any]]:
    """Solve multiple questions in parallel using multi-agent system."""
    orchestrator = MultiAgentOrchestrator(
        max_workers=max_workers,
        enable_checker=enable_checker
    )
    return orchestrator.solve_batch(questions)
