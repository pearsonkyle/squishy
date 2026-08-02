"""The loop must never delete a turn the model actually took.

This is the regression guard for the bug that motivated stripping the harness.
The old quality gate answered a "degenerate" tool call by appending a [system]
correction and skipping dispatch, which left the assistant's ``tool_calls``
message with no paired tool results. On the next turn ``normalize_messages``
repaired that orphan the only way it can — by deleting the assistant message
(tool-calling models emit empty ``content``, so there was nothing to downgrade
to) — and ``_merge_adjacent_same_role`` then glued the correction onto the task
statement itself.

The model was therefore told "you repeated the same command" while its history
contained no record of ever running one. The only coherent continuation is to
run it, which tripped the gate again: a loop generator wearing a loop breaker's
clothes.

Both halves are tested here: that repair still deletes an orphan (it must — the
endpoint 400s otherwise), and that the loop no longer manufactures one.
"""

from __future__ import annotations

import time

from conftest import FakeClient

from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.context import normalize_messages


def test_orphaned_assistant_turn_is_still_erased_by_repair():
    """The hazard is real: this is what the old gate handed to normalize."""
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "run_command", "arguments": '{"command": "grep -rn foo"}'},
        }]},
        {"role": "user", "content": "[system] You repeated the same command."},
    ]
    out = normalize_messages(msgs)

    assert not any(m.get("tool_calls") for m in out), (
        "an unpaired assistant tool_calls message cannot survive repair"
    )
    # And the correction lands on the task statement, not on its own turn.
    user_msgs = [m for m in out if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert "fix the bug" in user_msgs[0]["content"]
    assert "[system]" in user_msgs[0]["content"]


async def test_loop_never_produces_an_orphan(tmp_path):
    """Every tool call the model makes gets dispatched and paired.

    The model here repeats the *same* command over and over — precisely the
    shape the old gate would intercept. Nothing may be skipped: run_command's
    own echo counter answers inside the tool result, where the response is
    causally attached to the call that earned it.
    """
    repeat = [
        CompletionResult(
            text="",
            tool_calls=[ToolCall(id=f"c{i}", name="run_command",
                                 args={"command": "echo same"})],
        )
        for i in range(5)
    ]
    fake = FakeClient(script=[*repeat, CompletionResult(text="done.", tool_calls=[])])
    cfg = Config(
        working_dir=str(tmp_path), permission_mode="bench",
        max_turns=8, use_sandbox=False, save_sessions=False,
    )
    from squishy.agent import Agent

    agent = Agent(cfg, fake)  # type: ignore[arg-type]
    await agent.run("fix the bug")

    # Every list we actually sent upstream must be well-formed on its own —
    # normalize_messages should have had nothing to repair.
    for sent in fake.calls_seen:
        assert normalize_messages(sent) == sent, (
            "the loop handed normalize_messages a transcript it had to rewrite"
        )

    # And the model's own turns are all still there.
    assistant_turns = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistant_turns) >= 5


async def test_first_user_message_is_never_mutated(tmp_path):
    """The task statement is the anchor. Nothing may append to it."""
    task = "fix the bug in parser.py"
    fake = FakeClient(script=[
        CompletionResult(text="", tool_calls=[]),          # empty → nudged
        CompletionResult(text="all done", tool_calls=[]),
    ])
    cfg = Config(
        working_dir=str(tmp_path), permission_mode="bench",
        max_turns=5, use_sandbox=False, save_sessions=False,
    )
    from squishy.agent import Agent

    agent = Agent(cfg, fake)  # type: ignore[arg-type]
    await agent.run(task)

    first_user = next(m for m in agent.messages if m["role"] == "user")
    assert first_user["content"] == task


def test_loop_state_carries_no_gate_counters():
    """A structural guard against the gates growing back.

    Each removed gate needed its own counter on LoopState; a new one appearing
    here means feedback moved back out of the tool result and into the loop.
    """
    from squishy.agent_state import LoopState

    st = LoopState(start=time.monotonic())
    for gone in (
        "quality_retries", "quality_skips", "total_quality_violations",
        "phase", "explore_turns", "fix_verify_cycles", "shell_refusals",
        "plan_nudges", "auto_pytest_runs", "no_progress_intercepts",
        "f2p_finish_gate_intercepts", "env_fix_files",
    ):
        assert not hasattr(st, gone), f"{gone} came back"
