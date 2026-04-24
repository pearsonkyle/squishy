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
    # Build history with 2 prior identical read_file calls. The second-to-last
    # assistant message must be DIFFERENT so repeated_tool_call doesn't fire
    # first. The current turn's tool_calls are what we're assessing.
    messages = [
        _assistant_msg_with_calls([("read_file", {"path": "foo.py", "offset": 0, "limit": 100})]),
        {"role": "tool", "content": "file content..."},
        _assistant_msg_with_calls([("read_file", {"path": "foo.py", "offset": 0, "limit": 100})]),
        {"role": "tool", "content": "file content..."},
        # Second-to-last assistant msg uses a DIFFERENT tool so repeated_tool_call
        # doesn't match against the current turn's read_file call.
        _assistant_msg_with_calls([("edit_file", {"path": "bar.py", "old_str": "x", "new_str": "y"})]),
        {"role": "tool", "content": "ok"},
        # Current turn (most recent assistant msg — skipped by _extract_prev_tool_calls)
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
    assert "save_note" in msg


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


