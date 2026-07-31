"""v2 auto-pytest finish gate (intent tests).

When the agent edits in bench mode and then tries to finish without
ever running the FAIL_TO_PASS tests since that edit, the harness must
synthesize a pytest run, append the result to message history, and
inject a nudge so the agent gets one more shot to react.

These two tests encode the *why* of the feature (chance to react,
bounded loop). Behavior coverage like "non-bench mode skips" is
implicit in the guard clause and would only break if someone removed
the `permission_mode == "bench"` check.
"""
from __future__ import annotations

from conftest import FakeClient

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


_BENCH_PROMPT = """Fix the failing test.

## Failing Tests

- `test_target.py::test_truth`

## Problem

The function in src.py returns the wrong value.
"""


async def test_finish_without_pytest_triggers_auto_run(tmp_path):
    """The gate fires when the agent edits then prose-finishes without
    ever invoking pytest on the F2P tests.  Verify (a) auto-pytest
    actually ran (synthetic run_command appears in message history),
    and (b) the [system] nudge was injected so the agent gets another
    turn."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10
    cfg.max_auto_pytest_runs = 2

    # Seed a real failing test so pytest produces real output.
    #
    # Note: this test only asserts that the gate fires AT LEAST ONCE on
    # the first finish-without-pytest, not exactly once.  With cap=2 the
    # gate may fire again on a second prose finish; that's a separate
    # invariant covered by test_auto_run_capped_at_max_runs.
    (tmp_path / "src.py").write_text("def truth():\n    return False\n")
    (tmp_path / "test_target.py").write_text(
        "from src import truth\n\n"
        "def test_truth():\n"
        "    assert truth() is True\n"
    )

    fake = FakeClient(
        script=[
            # Turn 1: agent edits src.py (still wrong — returns 0 instead of True).
            CompletionResult(
                tool_calls=[
                    _tc("edit_file",
                        {"path": "src.py",
                         "old_str": "return False",
                         "new_str": "return 0"},
                        call_id="c1"),
                ]
            ),
            # Turn 2: agent tries to finish via prose without running pytest.
            #         Gate should fire here.
            CompletionResult(text="I think it's done.", tool_calls=[]),
            # Turn 3 (post-gate): agent gives up cleanly.
            CompletionResult(text="Actually no, can't fix it.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run(_BENCH_PROMPT)

    # Find the synthetic auto-pytest tool result in the message log.
    auto_results = [
        m for m in result.messages
        if m.get("role") == "tool"
        and isinstance(m.get("tool_call_id"), str)
        and str(m.get("tool_call_id", "")).startswith("auto-pytest-")
    ]
    assert len(auto_results) >= 1, (
        "expected the gate to fire at least once on first finish; "
        f"saw {len(auto_results)}"
    )
    # Pytest must actually have run — output should mention the test name
    # or a pytest framework token.
    body = str(auto_results[0].get("content", ""))
    assert (
        "test_truth" in body or "FAILED" in body
        or "test_target" in body or "pytest" in body.lower()
    ), f"auto-pytest produced no recognizable output: {body!r}"

    # The nudge should be in history right after the auto-pytest result.
    nudge_seen = any(
        m.get("role") == "user"
        and "without running the failing tests" in str(m.get("content", ""))
        for m in result.messages
    )
    assert nudge_seen, "auto-pytest nudge should have been injected"


async def test_auto_run_capped_at_max_runs(tmp_path):
    """The gate respects max_auto_pytest_runs.  With cap=1 and three
    consecutive finish attempts (each preceded by an edit so the gate
    keeps qualifying), the gate must fire at most once and the third
    finish must be allowed through."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10
    cfg.max_auto_pytest_runs = 1

    (tmp_path / "src.py").write_text(
        "def truth():\n    return False\n    # filler line a\n"
        "    # filler line b\n    # filler line c\n"
    )
    (tmp_path / "test_target.py").write_text(
        "from src import truth\n\n"
        "def test_truth():\n"
        "    assert truth() is True\n"
    )

    fake = FakeClient(
        script=[
            # Edit 1.
            CompletionResult(tool_calls=[
                _tc("edit_file",
                    {"path": "src.py",
                     "old_str": "    # filler line a",
                     "new_str": "    # changed a"},
                    call_id="c1"),
            ]),
            # Finish attempt 1 → gate fires (auto_pytest_runs: 0 → 1).
            CompletionResult(text="done 1.", tool_calls=[]),
            # Edit 2 (so gate would re-qualify if not capped).
            CompletionResult(tool_calls=[
                _tc("edit_file",
                    {"path": "src.py",
                     "old_str": "    # filler line b",
                     "new_str": "    # changed b"},
                    call_id="c2"),
            ]),
            # Finish attempt 2 → gate must NOT fire (cap reached); accepted.
            CompletionResult(text="done 2.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run(_BENCH_PROMPT)

    # Exactly one synthetic auto-pytest tool result (cap=1).
    auto_results = [
        m for m in result.messages
        if m.get("role") == "tool"
        and str(m.get("tool_call_id", "")).startswith("auto-pytest-")
    ]
    assert len(auto_results) == 1, (
        f"expected gate to fire exactly once (cap=1); saw {len(auto_results)}"
    )
    # The second finish attempt must have been accepted (final_text matches).
    assert result.success
    assert result.final_text == "done 2."
