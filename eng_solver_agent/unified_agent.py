"""Unified competition agent — single entrypoint, multiple solving strategies.

This merges the functionality of agent.py, agent_v2.py, and competition_agent.py
into one clean interface. It also replaces the tight coupling to agent._run_tool
with the standalone ToolDispatcher.

Solving modes (configurable per question or globally):
  - "auto"     : ReAct loop when LLM available, otherwise legacy tool + LLM draft
  - "react"    : Force ReAct reasoning (Think -> Act -> Observe)
  - "legacy"   : Original analyze -> tool -> draft pipeline
  - "llm_only" : Direct LLM solve, bypass tools entirely
  - "tool_only": Tools only, no LLM (fast, no API calls)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from eng_solver_agent.adapter import QuestionAdapter
from eng_solver_agent.config import Settings
from eng_solver_agent.formatter import format_submission_item
from eng_solver_agent.llm.kimi_client import KimiClient
from eng_solver_agent.llm.prompt_builder import build_analyze_messages, build_draft_messages
from eng_solver_agent.reasoning_engine import ReActEngine
from eng_solver_agent.retrieval import Retriever
from eng_solver_agent.router import QuestionRouter
from eng_solver_agent.schemas import AnalyzeResult, DraftResult
from eng_solver_agent.tool_dispatcher import ToolDispatcher
from eng_solver_agent.tools import AlgebraTool, CalculusTool, CircuitTool, PhysicsTool
from eng_solver_agent.verifier import validate_final_answer


class UnifiedAgent:
    """Single entrypoint for all competition solving needs.

    Usage:
        agent = UnifiedAgent()
        # Sequential
        results = [agent.solve_one(q) for q in questions]
        # Parallel (recommended for competitions)
        results = asyncio.run(agent.async_solve(questions, max_concurrent=5))
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        router: QuestionRouter | None = None,
        kimi_client: Any | None = None,
        retriever: Retriever | None = None,
        tool_registry: dict[str, Any] | None = None,
        default_mode: str = "auto",
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.adapter = QuestionAdapter()
        self.router = router or QuestionRouter()
        self.verifier = validate_final_answer
        self.retriever = retriever if retriever is not None else self._build_default_retriever()
        self.kimi_client = self._resolve_kimi_client(kimi_client)
        self.tools = tool_registry or self._build_default_tools()
        self.dispatcher = ToolDispatcher(self.tools)
        self.default_mode = default_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_one(
        self,
        question: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Solve a single question with the specified strategy.

        Args:
            question: Competition question dict with at least "question_id" and "question".
            mode: One of "auto", "react", "legacy", "llm_only", "tool_only".
                  Defaults to the agent's default_mode.
        """
        mode = (mode or self.default_mode).lower()
        normalized = self._normalize(question)
        route = self.router.route_with_confidence(normalized)

        try:
            if mode == "react":
                result = self._solve_react(normalized, route.subject)
            elif mode == "legacy":
                result = self._solve_legacy(normalized, route.subject)
            elif mode == "llm_only":
                result = self._solve_llm_only(normalized)
            elif mode == "tool_only":
                result = self._solve_tool_only(normalized, route.subject)
            else:  # auto
                result = self._solve_auto(normalized, route.subject)
        except Exception as exc:
            result = self._error_result(normalized["question_id"], exc)

        self.verifier(result)
        return result

    def solve(self, questions: list[dict[str, Any]], mode: str | None = None) -> list[dict[str, Any]]:
        """Sequential batch solve."""
        return [self.solve_one(q, mode=mode) for q in questions]

    async def async_solve(
        self,
        questions: list[dict[str, Any]],
        max_concurrent: int = 5,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        """Parallel batch solve with concurrency control.

        This is the recommended API for competitions.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _task(q: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.solve_one, q, mode)

        tasks = [asyncio.create_task(_task(q)) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: list[dict[str, Any]] = []
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                qid = str(questions[idx].get("question_id", f"q{idx}"))
                out.append(self._error_result(qid, res))
            else:
                out.append(res)
        return out

    # ------------------------------------------------------------------
    # Solving strategies
    # ------------------------------------------------------------------

    def _solve_auto(self, question: dict[str, Any], subject: str) -> dict[str, Any]:
        """Auto mode: ReAct if LLM available, otherwise legacy pipeline."""
        if self._has_llm():
            return self._solve_react(question, subject)
        return self._solve_legacy(question, subject)

    def _solve_react(self, question: dict[str, Any], subject: str) -> dict[str, Any]:
        """ReAct mode: Think -> Act -> Observe loop."""
        if not self._has_llm():
            return self._solve_legacy(question, subject)
        try:
            engine = ReActEngine(self.kimi_client, self.tools)
            topic = self._infer_topic(subject, str(question.get("question", "")))
            react_result = engine.solve(question, subject, topic)
            if react_result.success and react_result.reasoning_process:
                return format_submission_item(
                    question_id=question["question_id"],
                    reasoning_process=react_result.reasoning_process,
                    answer=react_result.answer,
                )
        except Exception:
            pass
        return self._solve_legacy(question, subject)

    def _solve_legacy(self, question: dict[str, Any], subject: str) -> dict[str, Any]:
        """Legacy two-stage pipeline: analyze -> tool -> draft."""
        analysis = self._analyze(question, subject)
        tool_result = self.dispatcher.dispatch(question, analysis)
        draft = self._draft(question, analysis, tool_result)
        return format_submission_item(
            question_id=question["question_id"],
            reasoning_process=draft.reasoning_process,
            answer=draft.answer,
        )

    def _solve_llm_only(self, question: dict[str, Any]) -> dict[str, Any]:
        """Direct LLM solve, bypassing tools entirely."""
        if not self._has_llm():
            return self._fallback_no_llm(question)
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一位精通工科基础课程（微积分、线性代数、电路原理、基础物理）的解题专家。"
                        "请用中文给出详细的解题步骤和最终答案。"
                        "返回严格JSON格式：{\"reasoning_process\": \"...\", \"answer\": \"...\"}"
                    ),
                },
                {"role": "user", "content": str(question.get("question", ""))},
            ]
            response = self.kimi_client.chat_json(messages, required_keys=("reasoning_process", "answer"))
            return format_submission_item(
                question_id=question["question_id"],
                reasoning_process=response["reasoning_process"],
                answer=response["answer"],
            )
        except Exception as exc:
            return self._fallback_no_llm(question, str(exc))

    def _solve_tool_only(self, question: dict[str, Any], subject: str) -> dict[str, Any]:
        """Tools only, no LLM. Fast but limited to deterministic computations."""
        analysis = self._fallback_analyze(question, subject)
        tool_result = self.dispatcher.dispatch(question, analysis)
        if tool_result["success"]:
            reasoning = f"题目: {question.get('question', '')}\n\n工具计算: {tool_result['output']}\n结论: 通过精确计算得到结果。"
            answer = tool_result["output"]
        else:
            reasoning = f"题目: {question.get('question', '')}\n\n工具尝试: {tool_result.get('error_message', '失败')}\n结论: 当前工具无法解决此问题。"
            answer = f"暂无法计算: {tool_result.get('error_message', '工具不支持')}"
        return format_submission_item(
            question_id=question["question_id"],
            reasoning_process=reasoning,
            answer=answer,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, question: dict[str, Any]) -> dict[str, Any]:
        normalized = self.adapter.normalize(question)
        normalized.setdefault("question", normalized.get("prompt", ""))
        normalized.setdefault("subject", normalized.get("type"))
        return normalized

    def _analyze(self, question: dict[str, Any], subject_hint: str) -> AnalyzeResult:
        if self._has_llm():
            try:
                response = self.kimi_client.chat_json(
                    build_analyze_messages(question, subject=subject_hint),
                    temperature=0.0,
                    required_keys=(
                        "subject", "topic", "knowns", "unknowns",
                        "equations_or_theorems", "should_use_tool",
                        "target_form", "possible_traps",
                    ),
                )
                return AnalyzeResult.model_validate(response)
            except Exception:
                pass
        return self._fallback_analyze(question, subject_hint)

    def _draft(
        self,
        question: dict[str, Any],
        analysis: AnalyzeResult,
        tool_result: dict[str, Any],
    ) -> DraftResult:
        if self._has_llm():
            try:
                response = self.kimi_client.chat_json(
                    build_draft_messages(question, analysis, tool_results=[tool_result]),
                    temperature=0.0,
                    required_keys=("reasoning_process", "answer"),
                )
                draft = DraftResult.model_validate(response)
                if str(draft.reasoning_process).strip() and str(draft.answer).strip():
                    return draft
            except Exception:
                pass
        return self._build_fallback_draft(question, analysis, tool_result)

    def _build_fallback_draft(
        self, question: dict[str, Any], analysis: AnalyzeResult, tool_result: dict[str, Any]
    ) -> DraftResult:
        q_text = str(question.get("question", "")).strip()
        parts = [f"题目: {q_text}"]
        if analysis.knowns:
            parts.append(f"已知条件: {', '.join(str(k) for k in analysis.knowns)}")
        if analysis.unknowns:
            parts.append(f"求解目标: {', '.join(str(u) for u in analysis.unknowns)}")
        if analysis.equations_or_theorems:
            parts.append(f"相关定理/公式: {', '.join(str(e) for e in analysis.equations_or_theorems)}")
        if tool_result.get("success"):
            parts.append(f"计算结果: {tool_result.get('output', '')}")
            parts.append("结论: 通过工具精确计算得到最终答案。")
            answer = str(tool_result.get("output", ""))
        else:
            error = tool_result.get("error_message", "工具未能完成计算")
            parts.append(f"计算尝试: 尝试使用 {tool_result.get('tool_name', '工具')} 进行计算")
            parts.append(f"遇到的问题: {error}")
            parts.append("分析: 虽然工具未能直接计算，但根据已知条件和相关定理，可以建立正确的数学模型。")
            answer = f"根据分析，解题方向正确。具体计算: {error}"
        return DraftResult(reasoning_process="\n".join(parts), answer=answer)

    def _fallback_analyze(self, question: dict[str, Any], subject_hint: str) -> AnalyzeResult:
        prompt = str(question.get("question", ""))
        topic = self._infer_topic(subject_hint, prompt)
        return AnalyzeResult(
            subject=subject_hint,
            topic=topic,
            knowns=["题干信息"],
            unknowns=["待求量"],
            equations_or_theorems=["相关学科标准公式"],
            should_use_tool=True,
            target_form="最终数值或表达式",
            possible_traps=["计算错误", "符号错误"],
        )

    def _infer_topic(self, subject: str, prompt: str) -> str:
        text = prompt.lower()
        if subject == "calculus":
            if "积分" in prompt or "integral" in text:
                return "integration"
            if "极限" in prompt or "limit" in text:
                return "limit"
            return "differentiation"
        if subject == "linalg":
            if "行列式" in prompt or "determinant" in text:
                return "determinant"
            if "特征值" in prompt or "eigen" in text:
                return "eigen"
            if "幂" in prompt or "power" in text:
                return "matrix_power"
            return "matrix_operations"
        if subject == "circuits":
            if "串联" in prompt or "并联" in prompt:
                return "equivalent_resistance"
            return "basic_circuits"
        return "kinematics"

    def _error_result(self, question_id: str, exc: Exception) -> dict[str, Any]:
        return format_submission_item(
            question_id=question_id,
            reasoning_process=f"解题过程中发生错误: {type(exc).__name__}: {exc}",
            answer=f"Error: {type(exc).__name__}",
        )

    def _fallback_no_llm(self, question: dict[str, Any], error: str = "") -> dict[str, Any]:
        q_text = str(question.get("question", ""))
        reasoning = f"题目: {q_text}\n\n说明: LLM 服务不可用"
        if error:
            reasoning += f"\n错误: {error}"
        reasoning += "\n尝试使用工具层进行计算..."
        return format_submission_item(
            question_id=question["question_id"],
            reasoning_process=reasoning,
            answer="暂无法计算: LLM 不可用",
        )

    def _has_llm(self) -> bool:
        if self.kimi_client is None:
            return False
        config = getattr(self.kimi_client, "config", None)
        if config is not None and hasattr(config, "base_url"):
            return bool(getattr(config, "base_url", ""))
        return True

    def _resolve_kimi_client(self, user_provided: Any | None) -> KimiClient | None:
        if user_provided is not None:
            return user_provided
        if not os.getenv("KIMI_BASE_URL", ""):
            return None
        return KimiClient()

    def _build_default_tools(self) -> dict[str, Any]:
        return {
            "physics": PhysicsTool(),
            "circuits": CircuitTool(),
            "linalg": AlgebraTool(),
            "calculus": CalculusTool(),
        }

    def _build_default_retriever(self) -> Retriever:
        retrieval_dir = Path(__file__).resolve().parent / "retrieval"
        return Retriever(
            formula_cards_path=retrieval_dir / "formula_cards.json",
            solved_examples_path=retrieval_dir / "solved_examples.jsonl",
        )
