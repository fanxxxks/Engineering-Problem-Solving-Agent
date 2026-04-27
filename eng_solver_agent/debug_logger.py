"""Verbose debug logger for the engineering solver agent.

Controls output via the AGENT_VERBOSE environment variable:
  - AGENT_VERBOSE=1  (default) : full output
  - AGENT_VERBOSE=0            : silent
  - AGENT_VERBOSE=2            : full output + raw HTTP payloads

On import, automatically sets up file logging (tee to terminal + txt file).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Read once at import time; can also be toggled at runtime via set_verbose()
_verbose_level = int(os.getenv("AGENT_VERBOSE", "1"))

# ── File logging (tee: terminal + file) ────────────────────────────────────

_log_file: Path | None = None
_original_stdout = sys.stdout
_file_logging_started = False


class _TeeWriter:
    """Duplicates all writes to both the terminal and a log file."""

    def __init__(self, log_path: str) -> None:
        self._file = open(log_path, "w", encoding="utf-8")
        self._terminal = _original_stdout

    @property
    def encoding(self) -> str:
        return getattr(self._terminal, "encoding", "utf-8")

    def write(self, text: str) -> None:
        try:
            self._terminal.write(text)
        except UnicodeEncodeError:
            try:
                self._terminal.write(text.encode(self._terminal.encoding or "utf-8", errors="replace").decode(self._terminal.encoding or "utf-8"))
            except Exception:
                pass
        clean = re.sub(r"\033\[[0-9;]*m", "", text)
        clean = clean.replace("\r", "")
        self._file.write(clean)

    def flush(self) -> None:
        try:
            self._terminal.flush()
        except Exception:
            pass
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def isatty(self) -> bool:
        return False


def start_file_logging(name: str = "agent_run") -> Path:
    """Begin duplicating all stdout to a timestamped log file in the project root.

    Returns the path to the log file.
    """
    global _log_file, _file_logging_started

    if _file_logging_started:
        return _log_file  # already started

    root = Path.cwd()
    ts = time.strftime("%Y%m%d_%H%M%S")
    _log_file = root / f"{name}_{ts}.txt"

    sys.stdout = _TeeWriter(str(_log_file))
    _file_logging_started = True
    # Re-check NO_COLOR now that we've replaced stdout
    global _NO_COLOR
    _NO_COLOR = not _original_stdout.isatty() or os.getenv("NO_COLOR")

    print(f"[log] 终端输出已同时保存至: {_log_file}")
    return _log_file


def stop_file_logging() -> None:
    """Restore original stdout and close the log file."""
    global _log_file, _file_logging_started
    if isinstance(sys.stdout, _TeeWriter):
        sys.stdout.close()
    sys.stdout = _original_stdout
    _log_file = None
    _file_logging_started = False


_start_file_logging = start_file_logging

def ensure_file_logging(name: str = "agent_run") -> Path:
    """Ensure file logging is started (idempotent)."""
    global _log_file, _file_logging_started
    if _file_logging_started:
        return _log_file
    return start_file_logging(name)

def set_verbose(level: int = 1) -> None:
    global _verbose_level
    _verbose_level = level


def is_verbose() -> bool:
    return _verbose_level >= 1


# ── ANSI colour helpers ────────────────────────────────────────────────────────

_NO_COLOR = not _original_stdout.isatty() or os.getenv("NO_COLOR")

_C = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "cyan":    "\033[96m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "red":     "\033[91m",
    "blue":    "\033[94m",
    "magenta": "\033[95m",
    "white":   "\033[97m",
    "gray":    "\033[90m",
}


def _c(text: str, color: str) -> str:
    if _NO_COLOR:
        return text
    return f"{_C.get(color, '')}{text}{_C['reset']}"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _truncate(text: str, limit: int = 800) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + _c(f" ...(+{len(text)-limit} chars)", "gray")


# ── Public API ─────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a prominent section divider."""
    if not is_verbose():
        return
    bar = "=" * 64
    print(_c(f"\n{bar}", "blue"), flush=True)
    print(_c(f"  {title}", "bold"), flush=True)
    print(_c(bar, "blue"), flush=True)


def step(module: str, message: str, color: str = "cyan") -> None:
    """Print a generic step message."""
    if not is_verbose():
        return
    print(_c(f"[{_ts()}][{module}]", color) + f" {message}", flush=True)


def log_llm_request(
    messages: list[dict[str, Any]],
    model: str = "",
    temperature: float | None = None,
    call_label: str = "LLM REQUEST",
) -> None:
    if not is_verbose():
        return
    print(_c(f"\n+-- {call_label} --- model={model}" + (f"  temperature={temperature}" if temperature is not None else ""), "blue"), flush=True)
    for msg in messages:
        role = msg.get("role", "?")
        content = str(msg.get("content", ""))
        role_color = {"system": "magenta", "user": "yellow", "assistant": "green"}.get(role, "white")
        print(_c(f"|  [{role.upper()}]", role_color))
        for line in _truncate(content, 600).splitlines():
            print(f"|    {line}")
    print(_c("+" + "-" * 63, "blue"), flush=True)


def log_llm_response(content: str, call_label: str = "LLM RESPONSE") -> None:
    if not is_verbose():
        return
    print(_c(f"\n+-- {call_label} -----------------------------------------------", "green"), flush=True)
    for line in _truncate(content, 1000).splitlines():
        print(f"|  {line}")
    print(_c("+" + "-" * 63, "green"), flush=True)


def log_llm_error(error: Exception, attempt: int, max_retry: int) -> None:
    if not is_verbose():
        return
    print(_c(f"[{_ts()}][LLM] 请求失败 (attempt {attempt}/{max_retry}): {type(error).__name__}: {error}", "red"), flush=True)


def log_json_parse(raw: str, success: bool, error: str = "") -> None:
    if not is_verbose():
        return
    if success:
        print(_c(f"[{_ts()}][LLM] JSON解析成功", "green"), flush=True)
    else:
        print(_c(f"[{_ts()}][LLM] JSON解析失败: {error}", "red"), flush=True)
        print(_c(f"  原始内容: {_truncate(raw, 400)}", "gray"), flush=True)


def log_route(subject: str, confidence: float, scores: dict[str, int], matched: tuple[str, ...]) -> None:
    if not is_verbose():
        return
    print(_c(f"\n[{_ts()}][ROUTER] 路由结果: ", "magenta") + _c(subject.upper(), "bold") + f"  confidence={confidence:.2f}", flush=True)
    print(_c(f"  各学科得分: {scores}", "gray"), flush=True)
    print(_c(f"  命中关键词: {list(matched)}", "gray"), flush=True)


def log_tool_dispatch(subject: str, operation: str, params: dict[str, Any]) -> None:
    if not is_verbose():
        return
    print(_c(f"\n[{_ts()}][TOOL] 调度工具: subject={subject}  operation={operation}", "yellow"), flush=True)
    for k, v in params.items():
        print(_c(f"  {k}: {_truncate(str(v), 200)}", "gray"), flush=True)


def log_tool_result(tool_name: str, success: bool, output: Any, error: str | None = None) -> None:
    if not is_verbose():
        return
    status = _c("[OK]", "green") if success else _c("[FAIL]", "red")
    print(_c(f"[{_ts()}][TOOL] {tool_name}  [{status}]", "yellow"), flush=True)
    if output:
        print(f"  输出: {_truncate(str(output), 400)}", flush=True)
    if error:
        print(_c(f"  错误: {error}", "red"), flush=True)


def log_react_step(step_num: int, thought: str, action: str | None, action_input: Any, observation: str | None) -> None:
    if not is_verbose():
        return
    print(_c(f"\n[{_ts()}][ReAct] ---- 步骤 {step_num} ----------------------------", "cyan"), flush=True)
    print(f"  思考: {_truncate(thought, 500)}", flush=True)
    if action:
        print(_c(f"  行动: {action}", "yellow") + (f"  输入: {action_input}" if action_input else ""), flush=True)
    if observation is not None:
        print(_c(f"  观察: {_truncate(str(observation), 300)}", "green"), flush=True)


def log_react_final(step_num: int, answer: str, success: bool) -> None:
    if not is_verbose():
        return
    label = _c("最终答案", "green") if success else _c("超出步数限制", "red")
    print(_c(f"\n[{_ts()}][ReAct] {label} (共{step_num}步)", "cyan"), flush=True)
    print(f"  答案: {_truncate(answer, 400)}", flush=True)


def log_pipeline_stage(stage: str, detail: str = "") -> None:
    if not is_verbose():
        return
    print(_c(f"\n[{_ts()}][PIPELINE] > {stage}", "blue") + (f"  {detail}" if detail else ""), flush=True)


def log_error(module: str, error: Exception) -> None:
    if not is_verbose():
        return
    print(_c(f"\n[{_ts()}][ERROR][{module}] {type(error).__name__}: {error}", "red"), flush=True)
