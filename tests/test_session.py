"""Tests for the session persistence module."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from squishy.session import (
    Session,
    append_messages,
    create_session,
    export_training,
    export_training_to_file,
    finish_session,
    list_sessions,
    load_messages,
    load_session,
    load_tools,
)


@pytest.fixture
def tmp_sessions(tmp_path: Path) -> str:
    """Return a temporary session root directory."""
    d = tmp_path / "sessions"
    d.mkdir()
    return str(d)


def test_create_and_load_session(tmp_sessions: str) -> None:
    sess = create_session(
        model="test-model",
        working_dir="/tmp/test",
        mode="yolo",
        root=tmp_sessions,
    )
    assert len(sess.id) == 32  # uuid4 hex
    assert sess.model == "test-model"
    assert sess.working_dir == "/tmp/test"
    assert sess.mode == "yolo"
    assert sess.status == "active"

    loaded = load_session(sess.id, root=tmp_sessions)
    assert loaded.id == sess.id
    assert loaded.model == "test-model"
    assert loaded.status == "active"


def test_append_and_load_messages(tmp_sessions: str) -> None:
    sess = create_session(model="m", working_dir="/tmp", mode="plan", root=tmp_sessions)

    # Append messages with tool_calls containing JSON string arguments
    # (as stored by agent.py) — should be normalized to dicts on write.
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "foo.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "name": "read_file", "content": "file content"},
    ]
    append_messages(sess.id, messages, root=tmp_sessions)

    loaded = load_messages(sess.id, root=tmp_sessions)
    assert len(loaded) == 4

    # Verify arguments were normalized from JSON string to dict.
    tc = loaded[2]["tool_calls"][0]
    assert isinstance(tc["function"]["arguments"], dict)
    assert tc["function"]["arguments"]["path"] == "foo.py"


def test_finish_session(tmp_sessions: str) -> None:
    sess = create_session(model="m", working_dir="/tmp", mode="yolo", root=tmp_sessions)
    finish_session(sess.id, status="completed", turns=5, tokens=1000, root=tmp_sessions)

    loaded = load_session(sess.id, root=tmp_sessions)
    assert loaded.status == "completed"
    assert loaded.turns == 5
    assert loaded.tokens == 1000


def test_list_sessions(tmp_sessions: str) -> None:
    # Create 3 sessions with different working dirs.
    s1 = create_session(model="m", working_dir="/a", mode="plan", root=tmp_sessions)
    s2 = create_session(model="m", working_dir="/b", mode="edits", root=tmp_sessions)
    s3 = create_session(model="m", working_dir="/a", mode="yolo", root=tmp_sessions)

    all_sessions = list_sessions(root=tmp_sessions)
    assert len(all_sessions) == 3

    # Filter by working_dir.
    filtered = list_sessions(working_dir="/a", root=tmp_sessions)
    assert len(filtered) == 2
    assert all(s.working_dir == "/a" for s in filtered)


def test_export_training_format(tmp_sessions: str) -> None:
    # Create session with tool schemas.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    sess = create_session(model="m", working_dir="/tmp", mode="yolo", tools=tools, root=tmp_sessions)

    # Append a conversation.
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": None,
            "think": "I should read the file first.",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "foo.py"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "name": "read_file", "content": "content"},
        {"role": "assistant", "content": "Fixed the bug by editing foo.py."},
    ]
    append_messages(sess.id, messages, root=tmp_sessions)

    # Export for training.
    record = export_training(sess.id, root=tmp_sessions)
    msgs = record["messages"]

    # System messages should be stripped.
    assert all(m["role"] != "system" for m in msgs)

    # First message should have tool schemas embedded.
    assert "tools" in msgs[0]
    assert msgs[0]["tools"][0]["function"]["name"] == "read_file"

    # Arguments should be dicts (already normalized on append).
    tc = msgs[1]["tool_calls"][0]
    assert isinstance(tc["function"]["arguments"], dict)

    # Think key should be preserved.
    assert msgs[1].get("think") == "I should read the file first."

    # Conversation should end with assistant message.
    assert msgs[-1]["role"] == "assistant"


def test_resume_loads_messages(tmp_sessions: str) -> None:
    sess = create_session(model="m", working_dir="/tmp", mode="plan", root=tmp_sessions)
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    append_messages(sess.id, messages, root=tmp_sessions)

    # Simulate resume: load messages from session.
    loaded = load_messages(sess.id, root=tmp_sessions)
    assert len(loaded) == 3
    assert loaded[0]["role"] == "system"
    assert loaded[2]["content"] == "hi there"


def test_session_dir_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_dir = str(tmp_path / "custom_sessions")
    monkeypatch.setenv("SQUISHY_SESSION_DIR", custom_dir)

    from squishy.session import session_dir
    d = session_dir()
    assert str(d) == custom_dir
    assert d.exists()


def test_export_to_file(tmp_sessions: str, tmp_path: Path) -> None:
    sess = create_session(model="m", working_dir="/tmp", mode="yolo", root=tmp_sessions)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    append_messages(sess.id, messages, root=tmp_sessions)

    out_path = tmp_path / "output.jsonl"
    result = export_training_to_file(sess.id, out_path, root=tmp_sessions)
    assert result == out_path
    assert out_path.exists()

    # Verify the output is valid JSONL with the expected structure.
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "messages" in record
    assert len(record["messages"]) == 2


def test_tools_persistence(tmp_sessions: str) -> None:
    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    sess = create_session(model="m", working_dir="/tmp", mode="yolo", tools=tools, root=tmp_sessions)

    loaded = load_tools(sess.id, root=tmp_sessions)
    assert len(loaded) == 1
    assert loaded[0]["function"]["name"] == "test_tool"


def test_multiple_appends(tmp_sessions: str) -> None:
    """Multiple append_messages calls should accumulate."""
    sess = create_session(model="m", working_dir="/tmp", mode="plan", root=tmp_sessions)
    append_messages(sess.id, [{"role": "user", "content": "first"}], root=tmp_sessions)
    append_messages(sess.id, [{"role": "assistant", "content": "response"}], root=tmp_sessions)
    append_messages(sess.id, [{"role": "user", "content": "second"}], root=tmp_sessions)

    loaded = load_messages(sess.id, root=tmp_sessions)
    assert len(loaded) == 3
    assert loaded[0]["content"] == "first"
    assert loaded[2]["content"] == "second"


def test_empty_append_noop(tmp_sessions: str) -> None:
    sess = create_session(model="m", working_dir="/tmp", mode="plan", root=tmp_sessions)
    append_messages(sess.id, [], root=tmp_sessions)

    # messages.jsonl should not exist (no messages were written).
    msgs_path = Path(tmp_sessions) / sess.id / "messages.jsonl"
    assert not msgs_path.exists()


def test_export_strips_trailing_tool_messages(tmp_sessions: str) -> None:
    """Export should strip trailing tool messages so conversation ends with assistant."""
    sess = create_session(model="m", working_dir="/tmp", mode="yolo", root=tmp_sessions)
    messages = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "data"},
    ]
    append_messages(sess.id, messages, root=tmp_sessions)

    record = export_training(sess.id, root=tmp_sessions)
    msgs = record["messages"]
    # The tool message at the end should be stripped.
    assert msgs[-1]["role"] == "assistant"
