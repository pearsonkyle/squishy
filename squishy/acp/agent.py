"""ACP agent implementation backed by squishy's Agent loop.

One ``SquishyAcpAgent`` instance is created per ``squishy-acp`` process and
spans the lifetime of the ACP connection. It maintains a registry of
``_AcpSession`` objects keyed by ACP ``sessionId`` — each session owns its
own ``squishy.agent.Agent`` (with persistent message history, plan state,
permission mode, and tool context).

Architecture
------------

Editor (ACP client)             SquishyAcpAgent             squishy.agent.Agent
─────────────────              ────────────────             ───────────────────
initialize        ─────────►   .initialize()    ─►   capabilities exchange
session/new       ─────────►   .new_session()   ─►   construct Agent + ToolContext
session/prompt    ─────────►   .prompt()        ─►   Agent.run(message, …)
                                                       │
                                                       │ Display ─►  AcpDisplay ─► session/update
                                                       │ prompt_fn ─► request_permission
                                                       │ ToolContext.fs_client ─► fs/read+write_text_file
                                                       │ ToolContext.terminal_client ─► terminal/*
                                                       ▼
                                                    TaskResult
                  ◄─────────  PromptResponse   ◄───  flush updates + map stop_reason
session/cancel    ─────────►   .cancel()       ─►   cancels the in-flight Agent.run()
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, replace
from typing import Any

from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    Implementation,
    InitializeResponse,
    McpCapabilities,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    SessionMode,
    SessionModeState,
    SetSessionModeResponse,
)

from squishy.acp.display import AcpDisplay
from squishy.acp.fs_bridge import AcpFsClient
from squishy.acp.permissions import make_prompt_fn
from squishy.acp.terminal_bridge import AcpTerminalClient
from squishy.agent import Agent
from squishy.client import Client
from squishy.config import INTERACTIVE_MODES, MODES, Config, PermissionMode
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.tool_restrictions import get_allowed_tools

log = logging.getLogger("squishy.acp.agent")

# ACP protocol version this implementation targets.
PROTOCOL_VERSION = 1

# Stop-reason values defined by the ACP schema.
_STOP_END = "end_turn"
_STOP_CANCELLED = "cancelled"
_STOP_REFUSAL = "refusal"


_MODE_DESCRIPTIONS: dict[str, str] = {
    "plan": "Read-only exploration. Edits and shell writes are blocked.",
    "edits": "File edits allowed; shell commands require approval.",
    "yolo": "Full autonomy — no per-tool approval prompts.",
    "bench": "Benchmark mode — phase-gated, action-biased.",
}


@dataclass
class _AcpSession:
    """Per-session state. One squishy Agent + the ACP capability bridges."""

    session_id: str
    cwd: str
    agent: Agent
    display: AcpDisplay
    permission_mode: PermissionMode = "plan"
    run_task: asyncio.Task[Any] | None = None


class SquishyAcpAgent:
    """ACP agent server backed by squishy.

    Construct with a base squishy ``Config`` (used as the template for every
    session — ``cwd`` is overridden per ``session/new``). Connection is
    established via ``acp.run_agent`` which calls ``on_connect`` with the
    client-side connection.
    """

    def __init__(self, base_config: Config) -> None:
        self._base_config = base_config
        self._client = Client(
            base_url=base_config.base_url,
            api_key=base_config.api_key,
            model=base_config.model,
            temperature=base_config.temperature,
            max_tokens=base_config.max_tokens,
            request_timeout=120.0,
            max_retries=base_config.max_consecutive_errors,
            thinking=base_config.thinking,
        )
        self._sessions: dict[str, _AcpSession] = {}
        self._conn: Any = None  # acp.Client, set in on_connect
        self._client_caps_fs_read = False
        self._client_caps_fs_write = False
        self._client_caps_terminal = False
        # Discovered after initialize so each session inherits them.

    # ── ACP lifecycle ─────────────────────────────────────────────────

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def aclose(self) -> None:
        await self._client.aclose()

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **_: Any,
    ) -> InitializeResponse:
        # Capture client capabilities so subsequent session/new calls
        # know whether to plug in the fs/terminal bridges.
        if client_capabilities is not None:
            fs = getattr(client_capabilities, "fs", None)
            if fs is not None:
                self._client_caps_fs_read = bool(getattr(fs, "read_text_file", False))
                self._client_caps_fs_write = bool(getattr(fs, "write_text_file", False))
            self._client_caps_terminal = bool(
                getattr(client_capabilities, "terminal", False),
            )

        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(
                    image=False, audio=False, embedded_context=True,
                ),
                mcp_capabilities=McpCapabilities(http=True, sse=True),
            ),
            agent_info=Implementation(name="squishy", version="0.2.0"),
        )

    async def authenticate(self, method_id: str, **_: Any) -> AuthenticateResponse | None:
        # Local LLMs don't need auth. Accept silently so editors that always
        # send authenticate don't choke.
        return None

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **_: Any,
    ) -> NewSessionResponse:
        # ``mcp_servers`` is accepted by the protocol but not yet merged
        # into squishy's MCP registry (which loads from .mcp.json on first
        # tool use). When we wire that in, this is the seam.
        session_id = f"sq-{os.urandom(8).hex()}"
        session = self._build_session(session_id, cwd)
        self._sessions[session_id] = session
        # Emit available commands + initial mode so the editor's chips render.
        self._emit_available_commands(session)
        return NewSessionResponse(
            session_id=session_id,
            modes=_mode_state(session.permission_mode),
        )

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        message_id: str | None = None,
        **_: Any,
    ) -> PromptResponse:
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(f"unknown session: {session_id}")

        text = _join_prompt_blocks(prompt)
        if not text.strip():
            return PromptResponse(stopReason=_STOP_END)

        loop = asyncio.get_running_loop()
        session.run_task = loop.create_task(session.agent.run(text))
        try:
            await session.run_task
            stop = _STOP_END
        except AgentCancelled:
            stop = _STOP_CANCELLED
        except (TimeoutError, AgentTimeout):
            stop = _STOP_END
        except LLMError as exc:
            log.warning("LLM error: %s", exc)
            stop = _STOP_REFUSAL
        except asyncio.CancelledError:
            stop = _STOP_CANCELLED
        finally:
            session.run_task = None
            await session.display.flush()
        return PromptResponse(stopReason=stop, userMessageId=message_id)

    async def cancel(self, session_id: str, **_: Any) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        if session.run_task is not None and not session.run_task.done():
            session.run_task.cancel()

    async def set_session_mode(
        self, mode_id: str, session_id: str, **_: Any,
    ) -> SetSessionModeResponse | None:
        session = self._sessions.get(session_id)
        if session is None or mode_id not in MODES:
            return SetSessionModeResponse()
        session.permission_mode = mode_id  # type: ignore[assignment]
        session.agent.config.permission_mode = mode_id  # type: ignore[assignment]
        session.agent.tool_ctx.permission_mode = mode_id
        session.display.set_mode(mode_id)
        self._emit_available_commands(session)
        return SetSessionModeResponse()

    # ── helpers ───────────────────────────────────────────────────────

    def _build_session(self, session_id: str, cwd: str) -> _AcpSession:
        cfg = _config_for_session(self._base_config, cwd)
        loop = asyncio.get_running_loop()
        display = AcpDisplay(self._conn, session_id, loop)
        prompt_fn = make_prompt_fn(self._conn, session_id)
        agent = Agent(
            cfg, self._client, display=display, prompt_fn=prompt_fn,
            session_id=session_id,
        )

        # Wire the per-session ACP filesystem + terminal bridges so the
        # built-in tools route through the editor when capabilities exist.
        if self._client_caps_fs_read or self._client_caps_fs_write:
            agent.tool_ctx.fs_client = AcpFsClient(
                self._conn, session_id,
                read=self._client_caps_fs_read,
                write=self._client_caps_fs_write,
            )
        if self._client_caps_terminal:
            agent.tool_ctx.terminal_client = AcpTerminalClient(
                self._conn, session_id,
            )

        return _AcpSession(
            session_id=session_id,
            cwd=cwd,
            agent=agent,
            display=display,
            permission_mode=cfg.permission_mode,
        )

    def _emit_available_commands(self, session: _AcpSession) -> None:
        allowed = sorted(get_allowed_tools(session.permission_mode))
        session.display.emit_available_commands(allowed)


# ── module-level helpers ──────────────────────────────────────────────


def _join_prompt_blocks(blocks: list[Any]) -> str:
    """Flatten an ACP prompt (list of content blocks) into a single string.

    Squishy's chat surface is text-only; non-text blocks are stringified so
    the model still gets some hint about their presence.
    """
    out: list[str] = []
    for b in blocks:
        kind = getattr(b, "type", None) or (
            b.get("type") if isinstance(b, dict) else None
        )
        if kind == "text":
            txt = getattr(b, "text", None) or (
                b.get("text", "") if isinstance(b, dict) else ""
            )
            out.append(str(txt))
        elif kind == "resource_link":
            uri = getattr(b, "uri", None) or (
                b.get("uri", "") if isinstance(b, dict) else ""
            )
            out.append(f"[link] {uri}")
        elif kind == "resource":
            resource = getattr(b, "resource", None) or (
                b.get("resource") if isinstance(b, dict) else None
            )
            if resource is not None:
                text = getattr(resource, "text", None) or (
                    resource.get("text", "") if isinstance(resource, dict) else ""
                )
                uri = getattr(resource, "uri", None) or (
                    resource.get("uri", "") if isinstance(resource, dict) else ""
                )
                if text:
                    out.append(f"--- {uri or 'resource'} ---\n{text}")
    return "\n\n".join(out).strip()


def _config_for_session(base: Config, cwd: str) -> Config:
    """Return a Config copy bound to *cwd* for a new ACP session."""
    return replace(base, working_dir=cwd)


def _mode_state(current: str) -> SessionModeState:
    modes = [
        SessionMode(id=m, name=m, description=_MODE_DESCRIPTIONS.get(m, m))
        for m in INTERACTIVE_MODES
    ]
    if current not in INTERACTIVE_MODES:
        current = "plan"
    return SessionModeState(available_modes=modes, current_mode_id=current)
