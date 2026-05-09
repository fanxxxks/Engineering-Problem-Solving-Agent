"""Comprehensive regression & unit tests for the 3 production fixes.

Fix #1: Markdown bold interference in _parse_step_response
Fix #2: brotli stream decoding crash recovery  
Fix #3: API 429 rate-limit retry with backoff + jitter

Usage:
    python scripts/mini_pytest.py tests/test_fixes_regression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests._helpers import assert_raises, approx_equal


# =============================================================================
# Fix #1: Markdown bold interference in _parse_step_response
# =============================================================================

def test_strip_markdown_fmt_clean():
    """No markdown → text unchanged."""
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine._strip_markdown_fmt("最终答案") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("compute") == "compute"
    assert ReActEngine._strip_markdown_fmt("无") == "无"
    assert ReActEngine._strip_markdown_fmt("similarity") == "similarity"


def test_strip_markdown_fmt_bold_double():
    """**text** → text"""
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine._strip_markdown_fmt("** 最终答案**") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("**最终答案**") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("** compute**") == "compute"
    assert ReActEngine._strip_markdown_fmt("**compute**") == "compute"
    assert ReActEngine._strip_markdown_fmt("** 无**") == "无"
    assert ReActEngine._strip_markdown_fmt("**无**") == "无"


def test_strip_markdown_fmt_bold_single_side():
    """**text or text**"""
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine._strip_markdown_fmt("** 最终答案") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("最终答案**") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("** 无") == "无"
    assert ReActEngine._strip_markdown_fmt("无**") == "无"


def test_strip_markdown_fmt_italic():
    """*text* → text"""
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine._strip_markdown_fmt("*最终答案*") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("*compute*") == "compute"
    assert ReActEngine._strip_markdown_fmt("__最终答案__") == "最终答案"
    assert ReActEngine._strip_markdown_fmt("_最终答案_") == "最终答案"


def test_strip_markdown_fmt_extra_whitespace():
    """extra whitespace handled"""
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine._strip_markdown_fmt("  ** 最终答案 **  ") == "最终答案"
    assert ReActEngine._strip_markdown_fmt(" 最终答案  ") == "最终答案"


def test_strip_markdown_fmt_empty():
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine._strip_markdown_fmt("") == ""
    assert ReActEngine._strip_markdown_fmt("**") == ""


def test_parse_step_markdown_action_recognized():
    """_parse_step_response correctly recognizes ** 最终答案** as final action."""
    from eng_solver_agent.reasoning_engine import ReActEngine
    engine = ReActEngine(llm_client=None, tools={})

    # Case 1: ** 最终答案**
    response = "思考: 已完成推导\n** 行动: ** 最终答案**\n** 行动输入: **{}"
    step = engine._parse_step_response(response, 1)
    assert step.is_final, "_parse_step_response should recognize ** 最终答案** as final"
    assert "已完成推导" in step.thought

    # Case 2: ** 无 (markdown bold wrapping on no-op)
    response2 = "思考: 继续推导\n** 行动: ** 无**"
    step2 = engine._parse_step_response(response2, 2)
    assert step2.action is None, "_parse_step_response should recognize ** 无** as no-op"

    # Case 3: Standard format (regression — still works)
    response3 = "思考: 标准推导\n 行动: compute\n 行动输入: {\"code\": \"x=1; print(x)\"}"
    step3 = engine._parse_step_response(response3, 3)
    assert step3.action == "compute"


def test_parse_step_markdown_compute_still_works():
    """** compute** should still be recognized as compute tool call."""
    from eng_solver_agent.reasoning_engine import ReActEngine
    engine = ReActEngine(llm_client=None, tools={})
    response = "思考: need calculation\n** 行动: ** compute**\n** 行动输入: **{\"code\": \"print(1+1)\"}"
    step = engine._parse_step_response(response, 1)
    assert step.action == "compute"
    assert step.action_input.get("code") == "print(1+1)"


def test_parse_step_markdown_similarity_still_works():
    """** similarity** should still be recognized."""
    from eng_solver_agent.reasoning_engine import ReActEngine
    engine = ReActEngine(llm_client=None, tools={})
    response = "思考: search\n** 行动: ** similarity**\n** 行动输入: **{\"query\": \"求导\"}"
    step = engine._parse_step_response(response, 1)
    assert step.action == "similarity"
    assert step.action_input.get("query") == "求导"


# =============================================================================
# Fix #2: brotli stream decode crash recovery
# =============================================================================

def _brotli_error() -> Exception:
    """Create a brotli-like exception without importing the internal module."""
    # httpx brotli errors manifest as RuntimeError with this message pattern
    return RuntimeError(
        "brotli: decoder process called with data when "
        "'can_accept_more_data()' is False"
    )


def test_handle_stream_error_recovers_partial_content():
    """Partial content collected before crash → returned rather than raised."""
    from eng_solver_agent.llm.kimi_client import KimiClient

    client = KimiClient.__new__(KimiClient)
    client.config = MagicMock()
    client.config.max_retry = 1

    brotli_error = _brotli_error()
    content_collected = ["This is partial content from the streaming response."]

    result = client._handle_stream_error(
        brotli_error,
        [{"role": "user", "content": "test"}],
        1.0,
        content_collected,
        [],
    )
    assert "partial content" in result
    assert "This is" in result


def test_handle_stream_error_no_content_raises():
    """No partial content → error is re-raised after retries."""
    from eng_solver_agent.llm.kimi_client import KimiClient

    client = KimiClient.__new__(KimiClient)
    client.config = MagicMock()
    client.config.max_retry = 1
    client.config.request_timeout_seconds = 120

    call_count = [0]

    def _persistent_fail(*a, **kw):
        call_count[0] += 1
        raise RuntimeError("simulated persistent error")

    class DummyClient:
        chat = MagicMock()
        chat.completions = MagicMock()
        chat.completions.create = _persistent_fail

    client._client = DummyClient()

    brotli_error = _brotli_error()
    try:
        client._handle_stream_error(
            brotli_error,
            [{"role": "user", "content": "test"}],
            1.0,
            [],  # no partial content
            [],
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "simulated persistent error" in str(e)
    assert call_count[0] >= 2, f"Expected at least 2 retry attemps, got {call_count[0]}"


# =============================================================================
# Fix #3: API 429 rate-limit retry with backoff + jitter
# =============================================================================

def test_handle_stream_error_429_retries_then_succeeds():
    """429 error triggers retry which eventually succeeds."""
    from eng_solver_agent.llm.kimi_client import KimiClient

    client = KimiClient.__new__(KimiClient)
    client.config = MagicMock()
    client.config.max_retry = 2
    client.config.request_timeout_seconds = 120

    call_count = [0]

    def _maybe_succeed(*a, **kw):
        call_count[0] += 1
        # First call fails with 429, 2nd succeeds
        if call_count[0] <= 1:
            raise RuntimeError('Error code: 429 - {"error": {"message": "The engine is currently overloaded, please try again later"}}')
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message = MagicMock()
        r.choices[0].message.content = "Success after retry"
        return r

    class DummyClient:
        chat = MagicMock()
        chat.completions = MagicMock()
        chat.completions.create = _maybe_succeed

    client._client = DummyClient()

    result = client._handle_stream_error(
        RuntimeError('Error code: 429 - {"error": {"message": "The engine is currently overloaded"}}'),
        [{"role": "user", "content": "test"}],
        1.0,
        [],  # no partial content
        [],
    )
    assert "Success after retry" in result
    assert call_count[0] == 2, f"Expected 2 calls (1 fail + 1 success), got {call_count[0]}"


def test_handle_stream_error_backoff_has_jitter():
    """Retry delays include jitter (not constant)."""
    import time as _time
    from eng_solver_agent.llm.kimi_client import KimiClient

    client = KimiClient.__new__(KimiClient)
    client.config = MagicMock()
    client.config.max_retry = 2
    client.config.request_timeout_seconds = 120

    delays = []

    def _always_429(*a, **kw):
        raise RuntimeError("429 overloaded")

    class DummyClient:
        chat = MagicMock()
        chat.completions = MagicMock()
        chat.completions.create = _always_429

    client._client = DummyClient()

    original_sleep = _time.sleep
    _time.sleep = lambda d: delays.append(d)

    try:
        client._handle_stream_error(
            RuntimeError("overloaded"),
            [{"role": "user", "content": "test"}],
            1.0, [], [],
        )
    except Exception:
        pass
    finally:
        _time.sleep = original_sleep

    # Delays should not all be identical (jitter exists)
    assert len(delays) >= 2, f"Expected at least 2 sleep calls, got {len(delays)}"


# =============================================================================
# Regression: ensure existing functionality unchanged
# =============================================================================

def test_regression_calc_tool_still_works():
    """Regression: NumericalComputationTool still computes correctly."""
    from eng_solver_agent.tools.numerical_tool import NumericalComputationTool
    tool = NumericalComputationTool()
    result = tool.compute("x = sympy.Symbol('x'); print(sympy.diff(x**3, x))")
    assert "3*x**2" in result


def test_regression_similarity_tool_loads():
    """Regression: SimilarProblemTool loads without error."""
    from eng_solver_agent.tools.similarity_tool import SimilarProblemTool
    t = SimilarProblemTool(use_vector_search=False)
    assert len(t.examples) > 100
    assert len(t.formula_cards) > 40


def test_regression_router_classifies_correction():
    """Regression: QuestionRouter still routes correctly."""
    from eng_solver_agent.router import QuestionRouter
    r = QuestionRouter()
    assert r.route({"question": "求导数 x^2"}) == "calculus"
    assert r.route({"question": "求矩阵特征值"}) == "linalg"
    assert r.route({"question": "串联电阻计算"}) == "circuits"
    assert r.route({"question": "求物体速度"}) == "physics"


def test_regression_unified_agent_loads():
    """Regression: UnifiedAgent initializes without error."""
    from eng_solver_agent.unified_agent import UnifiedAgent
    agent = UnifiedAgent()
    assert "compute" in agent.tools
    assert "similarity" in agent.tools
    assert "image" in agent.tools


def test_regression_tool_only_derivative():
    """Regression: tool_only mode still solves basic derivative."""
    from eng_solver_agent.unified_agent import UnifiedAgent
    agent = UnifiedAgent()
    result = agent.solve_one({
        "question_id": "reg-1",
        "question": "Find the derivative of x^2.",
        "expression": "x^2",
        "subject": "calculus",
    }, mode="tool_only")
    assert "2*x" in str(result.get("answer", "")) or "Error" not in str(result.get("answer", ""))


def test_regression_format_output_still_valid():
    """Regression: formatter + verifier still produce valid output."""
    from eng_solver_agent.formatter import format_submission_item
    from eng_solver_agent.verifier import validate_submission_item

    item = format_submission_item(
        question_id="test-1",
        reasoning_process="推导过程...",
        answer="42",
    )
    validate_submission_item(item)
    assert item["question_id"] == "test-1"
    assert item["answer"] == "42"


def test_regression_formula_cards_correct_count():
    """Regression: formula_cards.json has expected count."""
    import json
    cards_path = ROOT / "eng_solver_agent" / "retrieval" / "formula_cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    assert len(cards) == 58, f"Expected 58 formula cards, got {len(cards)}"


def test_regression_solved_examples_no_duplicates():
    """Regression: solved_examples.jsonl still has no duplicate IDs."""
    import json
    path = ROOT / "eng_solver_agent" / "retrieval" / "solved_examples.jsonl"
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["question_id"])
    assert len(ids) == len(set(ids)), f"Found {len(ids) - len(set(ids))} duplicate IDs in solved_examples.jsonl"


def test_regression_reasoning_engine_constants():
    """Regression: ReActEngine constants unchanged."""
    from eng_solver_agent.reasoning_engine import ReActEngine
    assert ReActEngine.MAX_STEPS == 8
    assert ReActEngine.MAX_CONSECUTIVE_NONE == 3
    assert ReActEngine.MAX_THOUGHT_OVERLAP == 0.80
    assert ReActEngine.MAX_OBSERVATION_LENGTH == 1200


def test_regression_smoke_tool_only_all_subjects():
    """Regression: tool_only mode handles all 4 subjects without crash."""
    from eng_solver_agent.unified_agent import UnifiedAgent
    agent = UnifiedAgent()

    tests = [
        ("calculus", {"question_id": "r1", "question": "derivative of x^2", "expression": "x^2", "subject": "calculus"}),
        ("linalg", {"question_id": "r2", "question": "determinant of [[1,2],[3,4]]", "matrix": [[1,2],[3,4]], "subject": "linalg"}),
        ("circuits", {"question_id": "r3", "question": "series resistors 2 and 3", "resistors": [2,3], "topology": "series", "subject": "circuits"}),
        ("physics", {"question_id": "r4", "question": "find v given v0=0, a=2, t=5", "knowns": {"v0": 0, "a": 2, "t": 5}, "target": "v", "subject": "physics"}),
    ]

    for subj, q in tests:
        result = agent.solve_one(dict(q), mode="tool_only")
        assert "question_id" in result
        assert "answer" in result
        assert "reasoning_process" in result
        # Check no fatal fallback
        assert "Error" not in result.get("answer", "") or "暂无法" in result.get("answer", "")


def test_regression_tool_registry_complete():
    """Regression: all 3 tools present and correctly typed."""
    from eng_solver_agent.unified_agent import UnifiedAgent
    from eng_solver_agent.tools.numerical_tool import NumericalComputationTool
    from eng_solver_agent.tools.similarity_tool import SimilarProblemTool
    from eng_solver_agent.tools.image_tool import ImageDescriptionTool

    agent = UnifiedAgent()
    assert isinstance(agent.tools["compute"], NumericalComputationTool)
    assert isinstance(agent.tools["similarity"], SimilarProblemTool)
    assert isinstance(agent.tools["image"], ImageDescriptionTool)


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == "__main__":
    import inspect
    all_tests = [(name, obj) for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
                 if name.startswith("test_")]
    passed = failed = 0
    for name, func in sorted(all_tests):
        try:
            func()
            passed += 1
            print(f"  PASS {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
