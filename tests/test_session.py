"""Tests for the session persistence module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from squishy.session import (
    append_messages,
    create_session,
    export_training,
    export_training_to_file,
    finish_session,
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




def test_finish_session(tmp_sessions: str) -> None:
    sess = create_session(model="m", working_dir="/tmp", mode="yolo", root=tmp_sessions)
    finish_session(sess.id, status="completed", turns=5, tokens=1000, root=tmp_sessions)

    loaded = load_session(sess.id, root=tmp_sessions)
    assert loaded.status == "completed"
    assert loaded.turns == 5
    assert loaded.tokens == 1000




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




