"""ReAct-style reasoning engine for competition-grade problem solving.

Architecture: Think -> Act -> Observe loop
- The LLM generates step-by-step reasoning in Chinese mathematical style
- Tool calls are embedded within the reasoning chain
- After each tool call, the result is fed back to the LLM
- The LLM can self-correct if a tool returns an error
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningStep:
    step_number: int
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_final: bool = False


@dataclass
class ReasoningResult:
    steps: list[ReasoningStep] = field(default_factory=list)
    reasoning_process: str = ""
    answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False


class ReActEngine:
    """ReAct reasoning engine that interleaves LLM reasoning with tool calls."""

    MAX_STEPS = 8

    def __init__(self, llm_client: Any, tools: dict[str, Any]) -> None:
        self.llm_client = llm_client
        self.tools = tools

    def solve(self, question: dict[str, Any], subject: str, topic: str) -> ReasoningResult:
        """Run the ReAct reasoning loop to solve a problem."""
        result = ReasoningResult()
        messages = self._build_initial_messages(question, subject, topic)

        for step_num in range(1, self.MAX_STEPS + 1):
            step = self._reason_step(messages, step_num)
            result.steps.append(step)

            if step.is_final:
                result.reasoning_process = self._format_reasoning_process(result.steps)
                result.answer = step.thought
                result.success = True
                return result

            if step.action and step.action != "none":
                observation = self._execute_action(step.action, step.action_input or {})
                step.observation = observation
                result.tool_calls.append({
                    "step": step_num,
                    "tool": step.action,
                    "input": step.action_input,
                    "output": observation,
                })
                feedback = (
                    f"【行动】调用工具: {step.action}({json.dumps(step.action_input or {}, ensure_ascii=False)})\n"
                    f"【观察】{observation}\n"
                    f"请继续下一步推理。"
                )
                messages.append({"role": "user", "content": feedback})
            else:
                messages.append({"role": "user", "content": f"【思考】{step.thought}\n请继续下一步推理。"})

        # Max steps reached
        result.reasoning_process = self._format_reasoning_process(result.steps)
        result.answer = result.steps[-1].thought if result.steps else "暂无法完成推理"
        result.success = False
        return result

    def _build_initial_messages(self, question: dict[str, Any], subject: str, topic: str) -> list[dict[str, str]]:
        q_text = str(question.get("question", ""))
        system_prompt = (
            "你是一位精通工科基础课程的解题专家。请使用中文，按照严谨的数学推导风格逐步解决以下问题。\n"
            "每一步都要有清晰的数学推导，使用标准数学符号和格式。\n"
            "需要计算时可以调用工具，最终给出明确的答案。\n"
            "推理过程要详细完整，用于评分。"
        )
        user_prompt = (
            f"【题目】{q_text}\n"
            f"【学科】{subject}\n"
            f"【知识点】{topic}\n\n"
            f"可用的工具：\n{self._describe_tools()}\n\n"
            f"请按照以下格式输出：\n"
            f"思考: [你的推理过程]\n"
            f"行动: [工具名称] 或 [无] 或 [最终答案]\n"
            f"行动输入: [工具参数JSON] 或 [空]\n\n"
            f"如果不需要工具，行动填'无'。\n"
            f"如果已经得到最终答案，行动填'最终答案'，思考中写出答案。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _describe_tools(self) -> str:
        descriptions = []
        for name, tool in self.tools.items():
            methods = [m for m in dir(tool) if not m.startswith("_") and callable(getattr(tool, m))]
            descriptions.append(f"- {name}: {', '.join(methods)}")
        return "\n".join(descriptions)

    def _reason_step(self, messages: list[dict[str, str]], step_num: int) -> ReasoningStep:
        """Ask the LLM for the next reasoning step."""
        if self.llm_client is None:
            return ReasoningStep(
                step_number=step_num,
                thought="LLM客户端不可用，无法继续推理。",
                action="最终答案",
                is_final=True,
            )

        try:
            # Request the next step from LLM
            step_messages = messages + [
                {"role": "user", "content": f"请输出第{step_num}步的思考、行动和行动输入："}
            ]
            response = self.llm_client.chat(step_messages)
            return self._parse_step_response(response, step_num)
        except Exception as exc:
            return ReasoningStep(
                step_number=step_num,
                thought=f"推理出错: {exc}",
                action="最终答案",
                is_final=True,
            )

    def _parse_step_response(self, response: str, step_num: int) -> ReasoningStep:
        """Parse the LLM response into a ReasoningStep."""
        thought_match = re.search(r"思考[:：]\s*(.+?)(?=\n行动[:：]|$)", response, re.DOTALL)
        action_match = re.search(r"行动[:：]\s*(.+?)(?=\n行动输入[:：]|$)", response, re.DOTALL)
        input_match = re.search(r"行动输入[:：]\s*(.+?)$", response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else response.strip()
        action = action_match.group(1).strip() if action_match else "无"
        action_input_str = input_match.group(1).strip() if input_match else "{}"

        if action in {"最终答案", "final_answer", "answer"}:
            return ReasoningStep(
                step_number=step_num,
                thought=thought,
                action="最终答案",
                is_final=True,
            )

        if action in {"无", "none", "None", ""}:
            return ReasoningStep(
                step_number=step_num,
                thought=thought,
                action=None,
            )

        # Parse action input as JSON
        action_input: dict[str, Any] = {}
        try:
            action_input = json.loads(action_input_str)
        except json.JSONDecodeError:
            action_input = {"raw_input": action_input_str}

        return ReasoningStep(
            step_number=step_num,
            thought=thought,
            action=action,
            action_input=action_input,
        )

    def _execute_action(self, action: str, action_input: dict[str, Any]) -> str:
        """Execute a tool action and return the observation."""
        tool = self.tools.get(action)
        if tool is None:
            return f"错误: 未找到工具 '{action}'"

        try:
            input_copy = dict(action_input)
            method_name = input_copy.pop("method", "solve")
            method = getattr(tool, method_name, None)
            if method is None:
                return f"错误: 工具 '{action}' 没有方法 '{method_name}'"

            result = method(**input_copy)
            return str(result)
        except Exception as exc:
            return f"错误: {type(exc).__name__}: {exc}"

    def _format_reasoning_process(self, steps: list[ReasoningStep]) -> str:
        """Format all steps into a competition-ready reasoning_process string."""
        parts: list[str] = []
        for step in steps:
            parts.append(f"步骤 {step.step_number}:")
            parts.append(f"  思考: {step.thought}")
            if step.action and step.action != "none":
                parts.append(f"  行动: {step.action}({json.dumps(step.action_input or {}, ensure_ascii=False)})")
            if step.observation:
                parts.append(f"  结果: {step.observation}")
            parts.append("")
        return "\n".join(parts)
