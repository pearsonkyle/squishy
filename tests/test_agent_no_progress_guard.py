"""F3: prose-completion no-progress guard.

When the agent prose-completes in bench/yolo mode after observing a
test failure via run_command but without making any edits to fix it,
the loop must intercept ONCE and nudge the agent to either fix or
explicitly call finish_plan(status="failure").

After one intercept the gate releases, so genuinely unfixable runs
can still terminate (otherwise install_status="degraded" instances
would loop forever).
"""
from __future__ import annotations

from squishy.agent import Agent, _has_unaddressed_test_failure
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display

from conftest import FakeClient


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


def test_helper_detects_assertion_failure():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "run_command",
            "content": "FAILED tests/test_x.py::test_y - AssertionError: expected 2",
        },
    ]
    summary = _has_unaddressed_test_failure(msgs)
    assert summary is not None
    assert "AssertionError" in summary or "FAILED" in summary


def test_helper_returns_none_when_edit_followed():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "run_command",
            "content": "FAILED tests/test_x.py - AssertionError",
        },
        {"role": "assistant", "tool_calls": [{"id": "c2"}]},
        {
            "role": "tool",
            "tool_call_id": "c2",
            "name": "edit_file",
            "content": "ok",
        },
    ]
    assert _has_unaddressed_test_failure(msgs) is None


def test_helper_returns_none_when_no_failure():
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "run_command",
            "content": "1 passed in 0.01s",
        },
    ]
    assert _has_unaddressed_test_failure(msgs) is None


def test_helper_detects_custom_repro_script_output():
    """The v27 scico bug: agent ran a repro script that printed
    'Adjoint test pass: False' but the failure-string heuristic must
    catch this exact phrasing too."""
    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "run_command",
            "content": "Adjoint test pass: False\nAdjoint accuracy: nan\n",
        },
    ]
    summary = _has_unaddressed_test_failure(msgs)
    assert summary is not None


async def test_agent_intercepts_prose_completion_after_failure(tmp_path):
    """End-to-end: the agent runs a failing test, then prose-completes.
    The first prose-completion is intercepted with a nudge; the second
    one is allowed through (single-shot)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 10

    # Pre-create a script that always exits non-zero so run_command
    # output contains a real failure string.
    fail_script = tmp_path / "fail.py"
    fail_script.write_text(
        "import sys\n"
        "print('FAILED tests/test_x.py - AssertionError: expected 2')\n"
        "sys.exit(1)\n"
    )

    fake = FakeClient(
        script=[
            # Turn 1: agent runs the failing script.
            CompletionResult(
                tool_calls=[
                    _tc("run_command",
                        {"command": "python fail.py"},
                        call_id="c1"),
                ]
            ),
            # Turn 2: agent prose-completes WITHOUT editing → intercepted.
            CompletionResult(text="I think it's done.", tool_calls=[]),
            # Turn 3: agent prose-completes again → released.
            CompletionResult(text="Really done.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("run fail.py")

    # The intercept should have fired exactly once.
    # We can't read LoopState directly post-run, but the script length
    # consumed (3 turns) confirms the loop continued past the first
    # prose-completion.
    assert fake._i == 3
    assert result.success
    assert result.final_text == "Really done."

    # The intercepted nudge should be in the message log.
    nudge_seen = any(
        m.get("role") == "user"
        and "did not call `edit_file`" in str(m.get("content", ""))
        for m in result.messages
    )
    assert nudge_seen, "no-progress nudge should have been injected"


async def test_agent_no_intercept_when_edit_followed(tmp_path):
    """Failure → edit → prose-complete should NOT trigger the gate."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 10

    fail_script = tmp_path / "fail.py"
    fail_script.write_text(
        "import sys\n"
        "print('FAILED - AssertionError')\n"
        "sys.exit(1)\n"
    )
    (tmp_path / "src.py").write_text("x = 1\n")

    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[
                    _tc("run_command",
                        {"command": "python fail.py"},
                        call_id="c1"),
                ]
            ),
            CompletionResult(
                tool_calls=[
                    _tc("edit_file",
                        {"path": "src.py",
                         "old_str": "x = 1",
                         "new_str": "x = 2"},
                        call_id="c2"),
                ]
            ),
            CompletionResult(text="Fixed.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert fake._i == 3
    assert result.success
    nudge_seen = any(
        m.get("role") == "user"
        and "did not call `edit_file`" in str(m.get("content", ""))
        for m in result.messages
    )
    assert not nudge_seen
