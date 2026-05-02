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
        """Run the ReAct reasoning loop to solve a problem.

        Adapts behaviour based on the question's difficulty field:
          - 简单 : LLM reasons directly, no tool calls, outputs final answer.
          - 中等 : Standard ReAct loop with compute / similarity tools.
          - 困难 : Pre-analyse step → structured plan, then standard ReAct.
        """
        difficulty = str(question.get("difficulty", "")).strip()
        log_pipeline_stage("ReAct 推理循环", f"subject={subject}, topic={topic}, difficulty={difficulty}, max_steps={self.MAX_STEPS}")
        result = ReasoningResult()
        messages = self._build_initial_messages(question, subject, topic, difficulty)

        # ── 困难: pre-analysis step ──────────────────────────────────────
        if difficulty == "困难":
            plan = self._pre_analyze_hard(question, subject, messages)
            if plan:
                messages.append({"role": "assistant", "content": f"【解题计划】\n{plan}"})
                messages.append({"role": "user", "content": (
                    "这是你刚才分析出的解题计划。"
                    "现在请从第 1 步开始，需要计算时调用 compute，需要参考时调用 similarity，"
                    "严格按计划逐步求解。"
                )})

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

            # After step 1: replace verbose user prompt with short instruction
            if step_num == 1 and len(messages) >= 2:
                messages[1]["content"] = (
                    f"题目：{str(question.get('question', ''))[:200]}\n"
                    f"不要反复推敲。不确定时立即 similarity 搜参考。\n"
                    f"【工具】只有三个，每个只有一种调用方式：\n\n"
                    f"  工具1 — compute（数值/符号计算）\n"
                    f"    行动: compute\n"
                    f"    行动输入: {{\"code\": \"Python代码\"}}\n"
                    f"    代码环境: sympy, sp, numpy, np, scipy, math, Symbol, solve, diff, integrate, limit, Matrix, pi, sin, cos, sqrt 等\n"
                    f"    必须用 print() 输出结果。\n\n"
                    f"  工具2 — similarity（相似题搜索，不确定时必须用！）\n"
                    f"    行动: similarity\n"
                    f"    行动输入: {{\"query\": \"完整题目文本\", \"subject\": \"{subject}\", \"topic\": \"{topic}\", \"top_k\": 5}}\n"
                    f"    query: 必填，完整题目或关键句。subject: 必填，只能填 physics / circuits / linalg / calculus。\n"
                    f"    topic: 选填，不确定就填 \"\"。top_k: 选填，默认 5。\n\n"
                    f"  工具3 — image（图片转文字，仅当题目有配图时使用）\n"
                    f"    行动: image\n"
                    f"    行动输入: {{\"image_path\": \"图片路径\"}}\n"
                    f"    将配图发送给视觉模型，返回结构化的电路/图表描述。\n\n"
                    f"得到答案后立即 最终答案。"
                )

            log_step_timing(f"ReAct 第{step_num}步", time.perf_counter() - t_step_start)

        # Max steps reached
        result.reasoning_process = self._format_reasoning_process(result.steps)
        result.answer = result.steps[-1].thought if result.steps else "暂无法完成推理"
        result.success = False
        log_react_final(step_num, result.answer, False)
        return result

    def _build_initial_messages(self, question: dict[str, Any], subject: str, topic: str, difficulty: str = "") -> list[dict[str, str]]:
        q_text = str(question.get("question", ""))
        difficulty = difficulty or str(question.get("difficulty", "")).strip()

        # Difficulty-specific system prompt additions
        if difficulty == "简单":
            difficulty_prompt = (
                "【本题难度：简单 — 禁止调用工具】\n"
                "此题可以直接求解。请在第 1 步推理完成后立即输出 行动: 最终答案。\n"
                "不要调用 compute、similarity 或 image 等任何工具。\n"
                "在思考中完成所有推导，在最终答案中给出明确的结果。\n\n"
            )
        elif difficulty == "困难":
            difficulty_prompt = (
                "【本题难度：困难 — 已生成解题计划】\n"
                "遇到不确定的步骤不要反复推敲，立即用 similarity 搜相似题参考。\n"
                "需要计算时用 compute。严格按计划逐步求解。\n\n"
            )
        else:
            difficulty_prompt = (
                "【本题难度：中等】\n"
                "先推导关键步骤，遇到不确定的不要反复推敲，用 similarity 搜参考。\n"
                "需要计算时用 compute 执行。\n\n"
            )

        system_prompt = (
            "你是一位精通工科基础课程的解题专家。请使用中文，按照严谨的数学推导风格逐步解决以下问题。\n"
            "每一步都要有清晰的数学推导，使用标准数学符号和格式。\n"
            "需要计算时可以调用工具，最终给出明确的答案。\n"
            "推理过程要详细完整，用于评分。\n\n"
        ) + difficulty_prompt + (
            "【重要】\n"
            "1. 不要反复推敲。不确定时立即调用 similarity。query 写完整题目，subject 从 physics/circuits/linalg/calculus 四选一，topic 不确定就填 \"\"。\n"
            "2. 得到答案后立即输出 行动: 最终答案，不要再用工具验证。\n"
            "3. 信任工具结果，工具调用一次就够。"
        )
        # If question has an image, include the path as a hint
        image_path = question.get("image", "")
        image_hint = f"【配图】{image_path}\n" if image_path else ""

        user_prompt = (
            f"【题目】{q_text}\n"
            f"{image_hint}"
            f"【学科】{subject}\n"
            f"【知识点】{topic}\n\n"
            f"【工具】只有三个，每个只有一种调用方式：\n\n"
            f"  工具1 — compute（数值/符号计算）\n"
            f"    行动: compute\n"
            f"    行动输入: {{\"code\": \"Python代码\"}}\n"
            f"    代码环境: sympy, sp, numpy, np, scipy, math, Symbol, solve, diff, integrate, limit, Matrix, pi, sin, cos, sqrt 等\n"
            f"    必须用 print() 输出结果。\n\n"
            f"  工具2 — similarity（相似题搜索，不确定时必须用！）\n"
            f"    行动: similarity\n"
            f"    行动输入: {{\"query\": \"完整题目文本\", \"subject\": \"{subject}\", \"topic\": \"{topic}\", \"top_k\": 5}}\n"
            f"    query: 必填，完整题目或关键句。subject: 必填，只能填 physics / circuits / linalg / calculus。\n"
            f"    topic: 选填，不确定就填 \"\"。top_k: 选填，默认 5。\n\n"
            f"  工具3 — image（图片转文字，仅当题目有配图时使用）\n"
            f"    行动: image\n"
            f"    行动输入: {{\"image_path\": \"图片路径\"}}\n"
            f"    将配图发送给视觉模型，返回结构化的电路/图表描述。\n\n"
            f"【输出格式】\n"
            f"  思考: [第一步判断难度（简单/中等/困难）和意图（学科/知识点/题型），然后推理]\n"
            f"  行动: compute | similarity | image | 无 | 最终答案\n"
            f"  行动输入: [JSON 或 {{}} ]\n\n"
            f"【行动详解】\n"
            f"  compute    → 需要数值/符号计算时使用，传入 Python 代码\n"
            f"  similarity → 不确定时搜相似题。query 写完整题目，subject 四选一，topic 不确定就空着\n"
            f"  image      → 题目有配图时，第一步就必须调用，传入图片路径获取描述\n"
            f"  无         → 本步纯推理，不需要调用任何工具，下一步继续推理\n"
            f"  最终答案   → 已经得到最终答案，在'思考'中写出完整答案后停止。\n"
            f"               如果之前已经调用过 compute 得到了计算结果，\n"
            f"               请在'思考'中汇总推理过程 + 工具计算结果 + 最终答案。\n"
            f"               得到答案后不要再用工具验证，直接结束。行动输入填 {{}}。"
        )
        user_content: str | list[dict[str, Any]] = user_prompt

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
                # Fallback to 'solve' if 'compute' doesn't exist
                method = getattr(tool, "solve", None)
                method_name = "solve"
            if method is None:
                return f"错误: 工具 '{action}' 没有方法 'compute' 或 'solve'"
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

    def _pre_analyze_hard(self, question: dict[str, Any], subject: str, messages: list[dict[str, Any]]) -> str | None:
        """For hard problems: ask the LLM to produce a structured step-by-step plan
        before entering the ReAct loop."""
        q_text = str(question.get("question", ""))
        plan_prompt = (
            f"你正在分析一道困难题。请按以下格式输出解题步骤规划：\n"
            f"1. [步骤名称]：需要做什么，用到什么公式/定理\n"
            f"2. [步骤名称]：...\n"
            f"...\n"
            f"最多输出 6 步。只输出步骤列表，不要写具体计算过程。\n\n"
            f"题目：{q_text}\n学科：{subject}"
        )
        plan_messages = [
            {"role": "system", "content": "你是一位工科解题专家。请用中文以 1. 2. 3. 4. 的格式列出解题步骤，每步一句话，简洁明了。"},
        ] + [m for m in messages if m["role"] in ("system",)] + [
            {"role": "user", "content": plan_prompt},
        ]
        try:
            response = self.llm_client.chat(plan_messages) if self.llm_client else None
        except Exception:
            response = None
        if isinstance(response, str) and response.strip():
            return response.strip()
        return None

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
