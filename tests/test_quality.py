"""Tests for the quality monitoring module."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from squishy.quality import assess_response, build_correction


@dataclass
class FakeToolCall:
    name: str
    args: dict[str, Any]
    id: str = "call_1"


def _assistant_msg_with_calls(calls: list[tuple[str, dict]]) -> dict:
    """Build an assistant message with tool_calls in the wire format."""
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, sort_keys=True, ensure_ascii=False),
                },
            }
            for i, (name, args) in enumerate(calls)
        ],
    }


# -- assess_response tests ---------------------------------------------------

def test_assess_ok():
    registry = {"read_file": object(), "edit_file": object()}
    tc = FakeToolCall(name="read_file", args={"path": "foo.py"})
    ok, reason = assess_response([tc], [], registry)
    assert ok is True
    assert reason == "ok"


def test_assess_unknown_tool():
    registry = {"read_file": object()}
    tc = FakeToolCall(name="find_file", args={})
    ok, reason = assess_response([tc], [], registry)
    assert ok is False
    assert reason == "unknown_tool:find_file"


def test_assess_malformed_args():
    registry = {"edit_file": object()}
    tc = FakeToolCall(name="edit_file", args={"_tool_arg_error": "bad JSON"})
    ok, reason = assess_response([tc], [], registry)
    assert ok is False
    assert reason == "malformed_args:edit_file"


def test_assess_repeated_tool_call():
    registry = {"read_file": object()}
    # The function skips the first (most recent) assistant msg in messages,
    # treating it as "the current turn". So we need the previous turn's
    # assistant msg PLUS a "current turn" assistant msg in the messages list.
    prev_msg = _assistant_msg_with_calls([("read_file", {"path": "foo.py"})])
    current_msg = _assistant_msg_with_calls([("read_file", {"path": "foo.py"})])
    messages = [prev_msg, {"role": "tool", "content": "ok"}, current_msg]
    tc = FakeToolCall(name="read_file", args={"path": "foo.py"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is False
    assert reason == "repeated_tool_call"


def test_assess_different_tool_call_ok():
    registry = {"read_file": object()}
    prev_msg = _assistant_msg_with_calls([("read_file", {"path": "foo.py"})])
    current_msg = _assistant_msg_with_calls([("read_file", {"path": "bar.py"})])
    messages = [prev_msg, {"role": "tool", "content": "ok"}, current_msg]
    tc = FakeToolCall(name="read_file", args={"path": "bar.py"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is True


def test_assess_excessive_reread():
    registry = {"read_file": object(), "edit_file": object()}
    # Build history with 2 prior identical read_file calls spread far enough
    # apart that repeated_tool_call (3-turn lookback) doesn't fire first.
    messages = [
        _assistant_msg_with_calls([("read_file", {"path": "foo.py", "offset": 0, "limit": 100})]),
        {"role": "tool", "content": "file content..."},
        _assistant_msg_with_calls([("edit_file", {"path": "a.py", "old_str": "x", "new_str": "y"})]),
        {"role": "tool", "content": "ok"},
        _assistant_msg_with_calls([("read_file", {"path": "foo.py", "offset": 0, "limit": 100})]),
        {"role": "tool", "content": "file content..."},
        _assistant_msg_with_calls([("edit_file", {"path": "b.py", "old_str": "x", "new_str": "y"})]),
        {"role": "tool", "content": "ok"},
        _assistant_msg_with_calls([("edit_file", {"path": "c.py", "old_str": "x", "new_str": "y"})]),
        {"role": "tool", "content": "ok"},
        _assistant_msg_with_calls([("edit_file", {"path": "d.py", "old_str": "x", "new_str": "y"})]),
        {"role": "tool", "content": "ok"},
        # Current turn (most recent — skipped by lookback)
        _assistant_msg_with_calls([("read_file", {"path": "foo.py", "offset": 0, "limit": 100})]),
    ]
    tc = FakeToolCall(name="read_file", args={"path": "foo.py", "offset": 0, "limit": 100})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is False
    assert reason == "excessive_reread"


def test_assess_empty_tool_calls():
    """Empty tool_calls list is OK (handled by existing empty-response logic)."""
    ok, reason = assess_response([], [], {})
    assert ok is True


# -- build_correction tests ---------------------------------------------------

def test_correction_unknown_tool():
    msg = build_correction("unknown_tool:find_file")
    assert "find_file" in msg
    assert "does not exist" in msg


def test_correction_repeated():
    msg = build_correction("repeated_tool_call")
    assert "stuck" in msg.lower() or "loop" in msg.lower()


def test_correction_malformed():
    msg = build_correction("malformed_args:edit_file")
    assert "edit_file" in msg
    assert "JSON" in msg


def test_correction_excessive_reread():
    msg = build_correction("excessive_reread")
    assert "edit_file" in msg
    assert "BLOCKED" in msg


def test_correction_edit_verify_loop():
    msg = build_correction("edit_verify_loop")
    assert "different approach" in msg.lower() or "cycling" in msg.lower()


# -- normalized command detection tests ----------------------------------------

def test_normalized_command_detection():
    """Variant flag ordering should be detected as the same command."""
    from squishy.quality import _normalize_command

    assert _normalize_command("pytest tests/foo.py -xvs") == _normalize_command(
        "pytest tests/foo.py -x -v -s"
    )
    assert _normalize_command("pytest tests/foo.py -xvs") == _normalize_command(
        "pytest tests/foo.py -s -v -x"
    )
    # Different positional args should NOT match.
    assert _normalize_command("pytest tests/foo.py -x") != _normalize_command(
        "pytest tests/bar.py -x"
    )


def test_repeated_command_with_normalized_flags():
    """_count_recent_commands should detect reordered flags as repeats."""
    from squishy.quality import _count_recent_commands

    messages = [
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/foo.py -xvs"})
        ]),
        {"role": "tool", "content": "ok"},
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/foo.py -x -v -s"})
        ]),
        {"role": "tool", "content": "ok"},
    ]
    count = _count_recent_commands(messages, "pytest tests/foo.py -svx", lookback=4)
    assert count >= 2


# -- edit-verify loop detection tests ------------------------------------------

def test_edit_verify_loop_detection():
    """Five consecutive edit->run_command cycles should trigger edit_verify_loop."""
    registry = {"edit_file": object(), "run_command": object()}
    messages: list[dict] = []
    # Build 5 consecutive cycles: edit_file then run_command
    for i in range(5):
        messages.append(_assistant_msg_with_calls([
            ("edit_file", {"path": "foo.py", "old_str": f"v{i}", "new_str": f"v{i+1}"})
        ]))
        messages.append({"role": "tool", "content": "ok"})
        messages.append(_assistant_msg_with_calls([
            ("run_command", {"command": f"pytest tests/test_foo.py attempt {i}"})
        ]))
        messages.append({"role": "tool", "content": "FAILED"})
    # Current turn: another edit
    messages.append(_assistant_msg_with_calls([
        ("edit_file", {"path": "foo.py", "old_str": "v5", "new_str": "v6"})
    ]))
    tc = FakeToolCall(name="edit_file", args={"path": "foo.py", "old_str": "v5", "new_str": "v6"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is False
    assert reason == "edit_verify_loop"


def test_edit_verify_loop_not_triggered_with_few_cycles():
    """Four cycles should NOT trigger the loop detector (threshold is 5)."""
    registry = {"edit_file": object(), "run_command": object()}
    messages: list[dict] = []
    for i in range(4):
        messages.append(_assistant_msg_with_calls([
            ("edit_file", {"path": "foo.py", "old_str": f"v{i}", "new_str": f"v{i+1}"})
        ]))
        messages.append({"role": "tool", "content": "ok"})
        messages.append(_assistant_msg_with_calls([
            ("run_command", {"command": f"pytest attempt {i}"})
        ]))
        messages.append({"role": "tool", "content": "FAILED"})
    messages.append(_assistant_msg_with_calls([
        ("edit_file", {"path": "foo.py", "old_str": "v4", "new_str": "v5"})
    ]))
    tc = FakeToolCall(name="edit_file", args={"path": "foo.py", "old_str": "v4", "new_str": "v5"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is True


def test_same_turn_duplicate_tool_calls():
    """Two identical tool calls in the same response should be caught as repeated_tool_call."""
    registry = {"read_file": object()}
    tc1 = FakeToolCall(name="read_file", args={"path": "foo.py"}, id="call_1")
    tc2 = FakeToolCall(name="read_file", args={"path": "foo.py"}, id="call_2")
    # Empty messages list — the duplicate-within-turn check fires before the history check.
    ok, reason = assess_response([tc1, tc2], [], registry)
    assert ok is False
    assert reason == "repeated_tool_call"


def test_edit_verify_cycle_with_read_only_in_between():
    """edit_file -> read_file -> run_command should count as one edit-verify cycle.

    The _count_edit_verify_cycles function skips read-only turns when looking
    ahead for a matching command turn, so the interleaved read should not break
    the streak.
    """
    from squishy.quality import _count_edit_verify_cycles

    # Build a sequence of 5 cycles each with a read_file between edit and run.
    messages: list[dict] = []
    for i in range(5):
        messages.append(_assistant_msg_with_calls([
            ("edit_file", {"path": "foo.py", "old_str": f"v{i}", "new_str": f"v{i+1}"})
        ]))
        messages.append({"role": "tool", "content": "ok"})
        # Read-only turn between edit and command — should be skipped.
        messages.append(_assistant_msg_with_calls([
            ("read_file", {"path": "foo.py"})
        ]))
        messages.append({"role": "tool", "content": "content"})
        messages.append(_assistant_msg_with_calls([
            ("run_command", {"command": f"pytest attempt {i}"})
        ]))
        messages.append({"role": "tool", "content": "FAILED"})

    count = _count_edit_verify_cycles(messages, lookback=20)
    # All 5 cycles should be counted even though reads are interleaved.
    assert count >= 5


def test_smart_cap_pytest_truncated_flag():
    """_smart_cap_pytest returns truncated=True when content was actually dropped."""
    from squishy.tools.shell import _smart_cap_pytest

    # Build a large pytest output that exceeds the cap.
    # Mix a failures section with lots of passing-output noise above it.
    passing_noise = "test_foo.py::test_a PASSED\n" * 500  # ~2700 chars of junk
    failures_block = (
        "= FAILURES =\n"
        "___ test_bad ___\n"
        "AssertionError: assert 1 == 2\n"
        "= 1 failed, 499 passed in 3.14s =\n"
    )
    raw = (passing_noise + failures_block).encode()

    cap = 200  # tight cap so we definitely truncate

    text, truncated = _smart_cap_pytest(raw, cap)
    assert truncated is True


# -- _count_edit_verify_cycles temporal tests ----------------------------------

# -- Post-edit test exemption tests -------------------------------------------

def test_run_command_not_repeated_after_edit():
    """run_command after edit_file should NOT be flagged as repeated_tool_call."""
    registry = {"read_file": object(), "edit_file": object(), "run_command": object()}
    messages = [
        # Turn 1: run_command (pre-edit test)
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/test_foo.py"})
        ]),
        {"role": "tool", "content": "FAILED"},
        # Turn 2: edit_file
        _assistant_msg_with_calls([
            ("edit_file", {"path": "foo.py", "old_str": "a", "new_str": "b"})
        ]),
        {"role": "tool", "content": "ok"},
        # Turn 3 (current): same run_command — should be allowed
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/test_foo.py"})
        ]),
    ]
    tc = FakeToolCall(name="run_command", args={"command": "pytest tests/test_foo.py"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is True, f"Expected ok=True but got {reason}"


def test_run_command_repeated_without_edit():
    """run_command without intervening edit should still be flagged."""
    registry = {"read_file": object(), "run_command": object()}
    messages = [
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/test_foo.py"})
        ]),
        {"role": "tool", "content": "FAILED"},
        # No edit between — just a read
        _assistant_msg_with_calls([
            ("read_file", {"path": "foo.py"})
        ]),
        {"role": "tool", "content": "content"},
        # Current turn: same run_command
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/test_foo.py"})
        ]),
    ]
    tc = FakeToolCall(name="run_command", args={"command": "pytest tests/test_foo.py"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is False
    assert reason in ("repeated_tool_call", "repeated_command")


def test_repeated_command_exempted_after_write_file():
    """write_file between two run_command calls should also exempt the repeat."""
    registry = {"write_file": object(), "run_command": object()}
    messages = [
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/"})
        ]),
        {"role": "tool", "content": "FAILED"},
        _assistant_msg_with_calls([
            ("write_file", {"path": "new.py", "content": "fix"})
        ]),
        {"role": "tool", "content": "ok"},
        _assistant_msg_with_calls([
            ("run_command", {"command": "pytest tests/"})
        ]),
    ]
    tc = FakeToolCall(name="run_command", args={"command": "pytest tests/"})
    ok, reason = assess_response([tc], messages, registry)
    assert ok is True, f"Expected ok=True but got {reason}"


def test_read_file_still_flagged_as_repeated():
    """read_file should still be flagged as repeated (not exempt like run_command)."""
    registry = {"read_file": object(), "edit_file": object()}
    prev_msg = _assistant_msg_with_calls([("read_file", {"path": "foo.py"})])
    current_msg = _assistant_msg_with_calls([("read_file", {"path": "foo.py"})])
    messages = [
        prev_msg,
        {"role": "tool", "content": "ok"},
        _assistant_msg_with_calls([
            ("edit_file", {"path": "bar.py", "old_str": "a", "new_str": "b"})
        ]),
        {"role": "tool", "content": "ok"},
        current_msg,
    ]
    tc = FakeToolCall(name="read_file", args={"path": "foo.py"})
    ok, reason = assess_response([tc], messages, registry)
    # read_file should still be caught by excessive_reread (check 4),
    # NOT exempted by the edit-between logic
    assert ok is False


def test_count_edit_verify_cycles_basic():
    """Three edit->test cycles should return 3."""
    from squishy.quality import _count_edit_verify_cycles

    messages: list[dict] = []
    for i in range(3):
        messages.append(_assistant_msg_with_calls([
            ("edit_file", {"path": "f.py", "old_str": f"v{i}", "new_str": f"v{i+1}"})
        ]))
        messages.append({"role": "tool", "content": "ok"})
        messages.append(_assistant_msg_with_calls([
            ("run_command", {"command": f"pytest tests/ attempt {i}"})
        ]))
        messages.append({"role": "tool", "content": "FAILED"})

    count = _count_edit_verify_cycles(messages, lookback=10)
    assert count == 3


def test_count_edit_verify_cycles_non_edit_breaks_streak():
    """A non-edit, non-read turn should break the streak."""
    from squishy.quality import _count_edit_verify_cycles

    messages: list[dict] = []
    # One cycle
    messages.append(_assistant_msg_with_calls([
        ("edit_file", {"path": "f.py", "old_str": "a", "new_str": "b"})
    ]))
    messages.append({"role": "tool", "content": "ok"})
    messages.append(_assistant_msg_with_calls([
        ("run_command", {"command": "pytest tests/"})
    ]))
    messages.append({"role": "tool", "content": "FAILED"})
    # Break with write_file (not read-only, not edit/run)
    messages.append(_assistant_msg_with_calls([
        ("write_file", {"path": "new.py", "content": "hello"})
    ]))
    messages.append({"role": "tool", "content": "ok"})
    # Another cycle
    messages.append(_assistant_msg_with_calls([
        ("edit_file", {"path": "f.py", "old_str": "b", "new_str": "c"})
    ]))
    messages.append({"role": "tool", "content": "ok"})
    messages.append(_assistant_msg_with_calls([
        ("run_command", {"command": "pytest tests/"})
    ]))
    messages.append({"role": "tool", "content": "FAILED"})

    count = _count_edit_verify_cycles(messages, lookback=20)
    # Should see only the most recent consecutive streak (1 cycle after the break)
    assert count <= 2  # streak breaks at write_file


