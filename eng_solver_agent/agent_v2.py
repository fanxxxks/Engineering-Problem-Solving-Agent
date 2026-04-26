"""Enhanced Agent with LLM-based reasoning and improved tool integration."""

from __future__ import annotations

import json
import os
from typing import Any

from eng_solver_agent.agent import EngineeringSolverAgent
from eng_solver_agent.debug_logger import log_pipeline_stage, section, step
from eng_solver_agent.llm.kimi_client import KimiClient
from eng_solver_agent.llm.prompt_builder import build_analyze_messages, build_draft_messages
from eng_solver_agent.config import Settings
from eng_solver_agent.formatter import format_submission_item
from eng_solver_agent.schemas import AnalyzeResult, DraftResult
from eng_solver_agent.verifier import validate_final_answer


class EnhancedSolverAgent(EngineeringSolverAgent):
    """Enhanced agent with better LLM integration and fallback reasoning."""

    def __init__(self, settings: Settings | None = None, **kwargs) -> None:
        super().__init__(settings, **kwargs)
        self.settings = settings or Settings.from_env()
        
    def solve_one(self, question: dict[str, Any]) -> dict[str, Any]:
        """Enhanced solving with LLM-first approach."""
        normalized = self._normalize_input(question)
        qid = normalized.get("question_id", "?")
        section(f"[开始] [EnhancedAgent] 开始解题: {qid}")
        step("EnhancedAgent", f"题目: {str(normalized.get('question', ''))[:120]}...")

        # Step 1: Use LLM to analyze and solve directly first
        log_pipeline_stage("尝试 LLM 直接求解")
        try:
            direct_result = self._llm_direct_solve(normalized)
            if direct_result and direct_result.get("answer"):
                result = format_submission_item(
                    question_id=normalized["question_id"],
                    reasoning_process=direct_result["reasoning"],
                    answer=direct_result["answer"],
                )
                validate_final_answer(result)
                return result
        except Exception as exc:
            step("EnhancedAgent", f"[失败] LLM 直接求解失败: {exc}", color="red")

        # Step 2: Fall back to original two-stage approach
        step("EnhancedAgent", "回退到传统两阶段模式", color="yellow")
        return super().solve_one(question)
    
    def _llm_direct_solve(self, question: dict[str, Any]) -> dict[str, str] | None:
        """Use LLM to directly solve the problem with detailed reasoning."""
        if not self._should_try_remote_client():
            return None
            
        question_text = question.get("question", "")
        
        # Build enhanced prompt with few-shot examples
        messages = [
            {
                "role": "system",
                "content": """You are an expert engineering problem solver. Solve the given problem step by step.

Instructions:
1. Analyze the problem carefully
2. Show your reasoning process clearly
3. Provide the final answer
4. Return your response in strict JSON format

Response format:
{
    "reasoning_process": "Detailed step-by-step solution process...",
    "answer": "Final numerical answer or expression"
}"""
            },
            {
                "role": "user",
                "content": f"Solve this problem:\n\n{question_text}"
            }
        ]
        
        try:
            response = self.kimi_client.chat_json(
                messages,
                required_keys=["reasoning_process", "answer"]
            )
            
            reasoning = response.get("reasoning_process", "").strip()
            answer = response.get("answer", "").strip()
            
            if reasoning and answer:
                return {
                    "reasoning": reasoning,
                    "answer": answer
                }
        except Exception:
            pass
            
        return None
    


# Backward compatibility
EngineeringSolverAgentV2 = EnhancedSolverAgent
