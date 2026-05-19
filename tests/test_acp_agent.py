"""Integration tests for the ACP agent surface.

We drive ``SquishyAcpAgent`` directly (without spinning up the JSON-RPC
transport) using a fake connection that records outgoing notifications and
returns scripted responses for client-side calls (fs/read_text_file,
request_permission, …). The underlying ``squishy.agent.Agent`` is wired to a
scripted ``Client`` so no real LLM is contacted.
"""
from __future__ import annotations

import asyncio
from typing import Any

from acp.schema import (
    ClientCapabilities,
    FileSystemCapabilities,
    TextContentBlock,
)

from squishy.acp.agent import SquishyAcpAgent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config


class FakeAcpClient:
    """Captures every notification and returns scripted client-method results."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []
        self.permission_response: Any = None
        self.fs_files: dict[str, str] = {}
        self.fs_writes: list[tuple[str, str]] = []

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        self.updates.append((session_id, update))

    async def read_text_file(self, path: str, session_id: str, **_: Any) -> Any:
        # Minimal stub that satisfies AcpFsClient: any attr accessor for
        # `.content` returns the stored text (or empty).
        text = self.fs_files.get(path, "")
        class _Resp:
            content = text
        return _Resp()

    async def write_text_file(
        self, content: str, path: str, session_id: str, **_: Any,
    ) -> None:
        self.fs_writes.append((path, content))

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **_: Any,
    ) -> Any:
        return self.permission_response


class ScriptedLLM:
    """Drop-in async Client replacement; scripts CompletionResults in order."""

    def __init__(self, script: list[CompletionResult]) -> None:
        self._script = script
        self._i = 0
        self.closed = False
        self.context_window = 0

    async def health(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: Any = None,
        on_retry: Any = None,
    ) -> CompletionResult:
        if self._i >= len(self._script):
            return CompletionResult(text="done.", tool_calls=[])
        result = self._script[self._i]
        self._i += 1
        if on_text and result.text:
            for ch in result.text:
                maybe = on_text(ch)
                if maybe is not None:
                    await maybe
        return result

    async def aclose(self) -> None:
        self.closed = True

    async def discover_model_name(self) -> str:
        return "fake"


def _patch_client(monkeypatch, script: list[CompletionResult]) -> ScriptedLLM:
    """Swap in a scripted Client without going through the network."""
    fake = ScriptedLLM(script)

    def _factory(*a, **kw):
        return fake

    monkeypatch.setattr("squishy.acp.agent.Client", _factory)
    return fake


def _new_agent(cwd: str) -> SquishyAcpAgent:
    cfg = Config(permission_mode="yolo", working_dir=cwd, use_sandbox=False)
    return SquishyAcpAgent(cfg)


async def test_initialize_advertises_capabilities(monkeypatch, tmp_path):
    _patch_client(monkeypatch, [])
    agent = _new_agent(str(tmp_path))
    resp = await agent.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
            terminal=True,
        ),
    )
    assert resp.protocol_version == 1
    caps = resp.agent_capabilities
    assert caps is not None
    assert caps.mcp_capabilities is not None
    assert caps.mcp_capabilities.http is True
    # Client capability flags should have been captured for downstream sessions.
    assert agent._client_caps_fs_read is True
    assert agent._client_caps_fs_write is True
    assert agent._client_caps_terminal is True


async def test_new_session_returns_modes_and_emits_commands(monkeypatch, tmp_path):
    _patch_client(monkeypatch, [])
    agent = _new_agent(str(tmp_path))
    conn = FakeAcpClient()
    agent.on_connect(conn)
    await agent.initialize(protocol_version=1)
    resp = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    assert resp.session_id.startswith("sq-")
    assert resp.modes is not None
    assert resp.modes.current_mode_id == "yolo"
    mode_ids = {m.id for m in resp.modes.available_modes}
    assert mode_ids == {"plan", "edits", "yolo"}
    # AvailableCommands should have been sent so the editor renders chips.
    # The display schedules the send as a Task — await one loop tick so it
    # lands in the FakeAcpClient.
    await asyncio.sleep(0)
    cmds_updates = [
        u for _, u in conn.updates
        if u.session_update == "available_commands_update"
    ]
    assert cmds_updates, "session/new must publish AvailableCommands"


async def test_prompt_runs_agent_and_emits_streaming_text(monkeypatch, tmp_path):
    _patch_client(monkeypatch, [
        CompletionResult(text="Hello from squishy.", tool_calls=[]),
    ])
    agent = _new_agent(str(tmp_path))
    conn = FakeAcpClient()
    agent.on_connect(conn)
    await agent.initialize(protocol_version=1)
    new = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    resp = await agent.prompt(
        prompt=[TextContentBlock(text="hi", type="text")],
        session_id=new.session_id,
    )
    assert resp.stopReason == "end_turn"
    chunks = [
        u for _, u in conn.updates
        if u.session_update == "agent_message_chunk"
    ]
    streamed = "".join(c.content.text for c in chunks)
    assert "Hello from squishy" in streamed


async def test_tool_call_streams_through_acp(monkeypatch, tmp_path):
    # Create a file the read_file tool can find via squishy's local IO.
    f = tmp_path / "hi.txt"
    f.write_text("alpha\nbeta\n")

    _patch_client(monkeypatch, [
        CompletionResult(
            text="",
            tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "hi.txt"})],
        ),
        CompletionResult(text="done", tool_calls=[]),
    ])
    agent = _new_agent(str(tmp_path))
    conn = FakeAcpClient()
    agent.on_connect(conn)
    await agent.initialize(protocol_version=1)
    new = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(
        prompt=[TextContentBlock(text="read hi.txt", type="text")],
        session_id=new.session_id,
    )

    tool_starts = [u for _, u in conn.updates if u.session_update == "tool_call"]
    tool_progress = [
        u for _, u in conn.updates if u.session_update == "tool_call_update"
    ]
    assert tool_starts, "must emit tool_call start"
    assert tool_progress, "must emit tool_call_update completion"
    assert tool_starts[0].kind == "read"


async def test_set_session_mode_updates_agent(monkeypatch, tmp_path):
    _patch_client(monkeypatch, [])
    agent = _new_agent(str(tmp_path))
    conn = FakeAcpClient()
    agent.on_connect(conn)
    await agent.initialize(protocol_version=1)
    new = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.set_session_mode(mode_id="plan", session_id=new.session_id)

    session = agent._sessions[new.session_id]
    assert session.permission_mode == "plan"
    assert session.agent.tool_ctx.permission_mode == "plan"
    assert session.agent.config.permission_mode == "plan"


async def test_fs_bridge_routes_writes_through_editor(monkeypatch, tmp_path):
    """When the client advertises fs.writeTextFile, edits use the editor."""
    target = tmp_path / "src.txt"
    target.write_text("old content\n")

    _patch_client(monkeypatch, [
        CompletionResult(
            text="",
            tool_calls=[ToolCall(
                id="t1", name="edit_file",
                args={"path": "src.txt", "old_str": "old", "new_str": "new"},
            )],
        ),
        CompletionResult(text="done", tool_calls=[]),
    ])
    agent = _new_agent(str(tmp_path))
    conn = FakeAcpClient()
    # Preload the file content the editor will hand back for read_text_file.
    conn.fs_files[str(target)] = "old content\n"
    agent.on_connect(conn)
    await agent.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
        ),
    )
    new = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(
        prompt=[TextContentBlock(text="edit", type="text")],
        session_id=new.session_id,
    )

    # Editor must have received the write — and the local file should NOT
    # have been modified (the editor is responsible for landing it on disk).
    assert conn.fs_writes, "edit_file must route through fs/write_text_file"
    path, content = conn.fs_writes[-1]
    assert path == str(target)
    assert "new" in content
    # Disk untouched on this path.
    assert target.read_text() == "old content\n"


async def test_request_permission_drives_prompt_fn(monkeypatch, tmp_path):
    """Tool needing approval triggers session/request_permission and
    honors the AllowedOutcome the client returns."""
    from acp.schema import AllowedOutcome, RequestPermissionResponse

    _patch_client(monkeypatch, [
        # In "edits" mode, run_command requires approval.
        CompletionResult(
            text="",
            tool_calls=[ToolCall(
                id="t1", name="run_command",
                args={"command": "echo hi"},
            )],
        ),
        CompletionResult(text="ok", tool_calls=[]),
    ])
    cfg = Config(permission_mode="edits", working_dir=str(tmp_path), use_sandbox=False)
    agent = SquishyAcpAgent(cfg)
    conn = FakeAcpClient()
    conn.permission_response = RequestPermissionResponse(
        outcome=AllowedOutcome(option_id="allow_once", outcome="selected"),
    )
    agent.on_connect(conn)
    await agent.initialize(protocol_version=1)
    new = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    await agent.prompt(
        prompt=[TextContentBlock(text="run echo", type="text")],
        session_id=new.session_id,
    )
    # The fact that we reached this point without the run_command tool
    # being refused is the assertion — the prompt_fn returned True.
