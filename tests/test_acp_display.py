"""Unit tests for the AcpDisplay adapter.

Verifies that every squishy Display method emits a structurally-correct
ACP session/update notification on the shared connection.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from squishy.acp.display import AcpDisplay


class FakeConn:
    """Stand-in for the agent-side acp.Client connection.

    Captures every (session_id, update) pair sent through session_update so
    tests can assert on the shape of the notifications squishy emits.
    """

    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        self.updates.append((session_id, update))


@pytest.fixture
async def display():
    loop = asyncio.get_running_loop()
    conn = FakeConn()
    d = AcpDisplay(conn, "sess-1", loop)
    yield d, conn
    await d.flush()


async def test_streaming_text_emits_agent_message_chunk(display):
    d, conn = display
    d.streaming_text_chunk("hello ")
    d.streaming_text_chunk("world")
    await d.flush()
    kinds = [u.session_update for _, u in conn.updates]
    assert kinds == ["agent_message_chunk", "agent_message_chunk"]
    texts = [u.content.text for _, u in conn.updates]
    assert texts == ["hello ", "world"]


async def test_tool_lifecycle_emits_start_and_progress(display):
    d, conn = display
    d.turn_header(1, 10, "read_file", "foo.py")
    d.tool_result(True, "42 lines loaded", 12.3)
    await d.flush()

    starts = [u for _, u in conn.updates if u.session_update == "tool_call"]
    progress = [u for _, u in conn.updates if u.session_update == "tool_call_update"]
    assert len(starts) == 1
    assert len(progress) == 1
    assert starts[0].title.startswith("read_file")
    assert starts[0].kind == "read"
    assert starts[0].status == "in_progress"
    # The progress event must carry the same tool_call_id so the editor
    # correlates the completion with the right card.
    assert progress[0].tool_call_id == starts[0].tool_call_id
    assert progress[0].status == "completed"


async def test_edit_diff_attaches_file_edit_content(display):
    d, conn = display
    d.turn_header(1, 10, "edit_file", "src/a.py")
    d.edit_diff("src/a.py", "old line", "new line")
    await d.flush()

    edits = [
        u for _, u in conn.updates
        if u.session_update == "tool_call_update" and u.content
    ]
    assert edits
    block = edits[0].content[0]
    assert block.type == "diff"
    assert block.path == "src/a.py"
    assert block.new_text == "new line"
    assert block.old_text == "old line"


async def test_plan_progress_emits_plan_update(display):
    d, conn = display
    d.plan_progress([
        {"description": "explore", "status": "done", "note": ""},
        {"description": "fix bug", "status": "in-progress", "note": ""},
        {"description": "verify", "status": "pending", "note": ""},
    ])
    await d.flush()

    plans = [u for _, u in conn.updates if u.session_update == "plan"]
    assert len(plans) == 1
    statuses = [e.status for e in plans[0].entries]
    assert statuses == ["completed", "in_progress", "pending"]


async def test_mode_changed_emits_current_mode_update(display):
    d, conn = display
    d.mode_changed("yolo")
    await d.flush()

    modes = [u for _, u in conn.updates if u.session_update == "current_mode_update"]
    assert len(modes) == 1
    assert modes[0].current_mode_id == "yolo"


async def test_available_commands_carries_tool_names(display):
    d, conn = display
    d.emit_available_commands(["read_file", "edit_file"])
    await d.flush()

    cmds_updates = [
        u for _, u in conn.updates
        if u.session_update == "available_commands_update"
    ]
    assert len(cmds_updates) == 1
    names = [c.name for c in cmds_updates[0].available_commands]
    assert names == ["read_file", "edit_file"]


async def test_unknown_method_is_absorbed(display):
    d, _ = display
    # squishy.display.Display has many methods (banner, summary, status, …) the
    # adapter doesn't override. The __getattr__ fallback should swallow them.
    d.banner("http://x", "model-y")
    d.summary(3, 1.2)
    d.status("plan")
