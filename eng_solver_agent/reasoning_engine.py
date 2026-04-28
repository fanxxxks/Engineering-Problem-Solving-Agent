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
import time
from dataclasses import dataclass, field
from typing import Any

from eng_solver_agent.debug_logger import log_react_step, log_react_final, log_pipeline_stage, log_step_timing, step


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
        # Build a reverse lookup map: method_name -> (tool_name, tool_instance)
        self._method_registry: dict[str, tuple[str, Any]] = {}
        for tool_name, tool in tools.items():
            for attr_name in dir(tool):
                if not attr_name.startswith("_") and callable(getattr(tool, attr_name, None)):
                    # Register method -> (tool_name, tool); first one wins on collision
                    if attr_name not in self._method_registry:
                        self._method_registry[attr_name] = (tool_name, tool)

    def solve(self, question: dict[str, Any], subject: str, topic: str) -> ReasoningResult:
        """Run the ReAct reasoning loop to solve a problem."""
        log_pipeline_stage("ReAct 推理循环", f"subject={subject}, topic={topic}, max_steps={self.MAX_STEPS}")
        result = ReasoningResult()
        messages = self._build_initial_messages(question, subject, topic)

        for step_num in range(1, self.MAX_STEPS + 1):
            t_step_start = time.perf_counter()
            step("ReActEngine", f"[循环] ReAct 第 {step_num}/{self.MAX_STEPS} 步...", color="cyan")
            rs = self._reason_step(messages, step_num)
            result.steps.append(rs)

            if rs.is_final:
                result.reasoning_process = self._format_reasoning_process(result.steps)
                result.answer = rs.thought
                result.success = True
                log_react_final(step_num, rs.thought, True)
                log_step_timing(f"ReAct 第{step_num}步 (最终答案)", time.perf_counter() - t_step_start)
                return result

            if rs.action and rs.action != "none":
                log_react_step(step_num, rs.thought, rs.action, rs.action_input, None)
                observation = self._execute_action(rs.action, rs.action_input or {})
                rs.observation = observation
                log_react_step(step_num, rs.thought, rs.action, rs.action_input, observation)
                result.tool_calls.append({
                    "step": step_num,
                    "tool": rs.action,
                    "input": rs.action_input,
                    "output": observation,
                })
                feedback = (
                    f"【行动】调用工具: {rs.action}({json.dumps(rs.action_input or {}, ensure_ascii=False)})\n"
                    f"【观察】{observation}\n"
                    f"请继续下一步推理。"
                )
                messages.append({"role": "user", "content": feedback})
            else:
                log_react_step(step_num, rs.thought, None, None, None)
                messages.append({"role": "user", "content": f"【思考】{rs.thought}\n请继续下一步推理。"})

            log_step_timing(f"ReAct 第{step_num}步", time.perf_counter() - t_step_start)

        # Max steps reached
        result.reasoning_process = self._format_reasoning_process(result.steps)
        result.answer = result.steps[-1].thought if result.steps else "暂无法完成推理"
        result.success = False
        log_react_final(step_num, result.answer, False)
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
            f"可用的工具及调用方式（请严格按照以下格式调用）：\n{self._describe_tools()}\n\n"
            f"请按照以下格式输出：\n"
            f"思考: [你的推理过程]\n"
            f"行动: [工具名称] 或 [无] 或 [最终答案]\n"
            f"行动输入: [工具参数JSON] 或 [空]\n\n"
            f"规则说明：\n"
            f"1. 工具名称必须是上方列出的工具名（如 compute / similarity），不要加命名空间前缀。\n"
            f"2. 调用 compute 时，在'行动输入'中传入 JSON：{{\"code\": \"你的Python代码\"}}。\n"
            f"   代码可使用 sympy/sp 符号运算、numpy/np 数值计算、scipy 科学计算、math 数学函数。\n"
            f"   使用 print() 输出最终结果，系统会捕获并返回。\n"
            f"   示例：{{\"code\": \"x = sympy.Symbol('x'); result = sympy.diff(x**2, x); print(result)\"}}\n"
            f"3. 调用 similarity 时，在'行动输入'中传入 JSON：{{\"query\": \"搜索内容\"}} 或 {{\"method\": \"find_similar\", ...}}。\n"
            f"4. 行动输入必须是合法的 JSON 对象。\n"
            f"5. 如果不需要工具，行动填'无'。\n"
            f"6. 如果已经得到最终答案，行动填'最终答案'，思考中写出答案。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    _METHOD_SIGNATURES: dict[str, str] = {
        "solve": 'solve(code="Python code using sympy/numpy/scipy/math, use print() to output")',
        "compute": 'compute(code="Python code using sympy/numpy/scipy/math, use print() to output")',
        "compute_from_query": 'compute_from_query(code="Python code using sympy/numpy/scipy/math")',
    }

    _PARAM_ALIASES: dict[str, dict[str, str]] = {
        "solve": {"query": "code", "expression": "code", "expr": "code"},
        "compute": {"query": "code", "expression": "code", "expr": "code"},
        "compute_from_query": {"query": "code", "expression": "code", "expr": "code"},
    }

    @staticmethod
    def _normalize_action_input(method_name: str, action_input: dict[str, Any]) -> dict[str, Any]:
        aliases = ReActEngine._PARAM_ALIASES.get(method_name, {})
        if not aliases:
            return dict(action_input)
        result = dict(action_input)
        for bad_name, good_name in aliases.items():
            if bad_name in result and good_name not in result:
                result[good_name] = result.pop(bad_name)
        return result

    def _describe_tools(self) -> str:
        """Return a human-readable description of all available tools with parameter signatures."""
        lines: list[str] = []
        for name, tool in self.tools.items():
            doc = (tool.__doc__ or "").split("\n")[0].strip()
            methods = [m for m in dir(tool) if not m.startswith("_") and callable(getattr(tool, m, None))]
            lines.append(f"- {name}: {doc}")
            lines.append(f"  可用方法: {', '.join(methods)}")
        lines.append("")
        lines.append("常用方法及参数签名（可直接作为'行动'使用）：")
        seen: set[str] = set()
        for method, (tool_name, _) in self._method_registry.items():
            if method not in seen:
                seen.add(method)
                sig = ReActEngine._METHOD_SIGNATURES.get(method, f"{method}(...)")
                lines.append(f"  {sig} -> {tool_name}")
        return "\n".join(lines)

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
                thought=f"推理步骤出错: {exc}",
                action=None,
                is_final=False,
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
        """Execute a tool action and return the observation.

        Supports multiple calling conventions:
        1. action="tool_name.method_name"
        2. action="method_name" → lookup in _method_registry
        3. action="tool_name" → use method from action_input["method"]
        4. action="compute" or "solve" → route to NumericalComputationTool
        """
        action = action.strip()

        def _call_method(method, m_name: str, inputs: dict[str, Any]) -> str:
            normalized = self._normalize_action_input(m_name, inputs)
            return str(method(**normalized))

        # 1: "tool.method" syntax
        if "." in action:
            parts = action.split(".", 1)
            tool_name, method_name = parts[0].strip(), parts[1].strip()
            tool = self.tools.get(tool_name)
            if tool is None:
                return f"错误: 未找到工具 '{tool_name}'"
            method = getattr(tool, method_name, None)
            if method is None:
                return f"错误: 工具 '{tool_name}' 没有方法 '{method_name}'"
            try:
                return _call_method(method, method_name, action_input)
            except Exception as exc:
                return f"错误: {type(exc).__name__}: {exc}"

        # 2: action is a known tool name
        tool = self.tools.get(action)
        if tool is not None:
            input_copy = dict(action_input)
            method_name = input_copy.pop("method", "compute")
            method = getattr(tool, method_name, None)
            if method is None:
                return f"错误: 工具 '{action}' 没有方法 '{method_name}'"
            try:
                return _call_method(method, method_name, input_copy)
            except Exception as exc:
                return f"错误: {type(exc).__name__}: {exc}"

        # 3: action is a method name → find the tool that has it
        if action in self._method_registry:
            tool_name, tool = self._method_registry[action]
            method = getattr(tool, action, None)
            if method is not None:
                try:
                    return _call_method(method, action, action_input)
                except Exception as exc:
                    return f"错误: {type(exc).__name__}: {exc}"

        # 4: special shortcuts for compute / solve
        if action in ("compute", "solve"):
            for name, tool in self.tools.items():
                if hasattr(tool, action):
                    method = getattr(tool, action)
                    try:
                        return _call_method(method, action, action_input)
                    except Exception as exc:
                        return f"错误: {type(exc).__name__}: {exc}"

        return (
            f"错误: 未找到工具或方法 '{action}'。"
            f"可用工具: {list(self.tools.keys())}。"
            f"可用快捷方法: {list(self._method_registry.keys())[:20]}..."
        )

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
