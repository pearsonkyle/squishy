"""Adapter that turns the ACP ``Client`` terminal methods into the small
``create``/``output``/``wait_for_exit``/``kill``/``release`` surface squishy's
``run_command`` tool expects.

When an ACP client advertises ``terminal``, ``run_command`` routes through
this bridge so the command shows up in the editor's terminal pane and its
output streams to the user as it happens.
"""
from __future__ import annotations

from typing import Any


class AcpTerminalClient:
    """Thin wrapper around the agent-side ACP connection for terminal IO."""

    def __init__(self, conn: Any, session_id: str) -> None:
        self._conn = conn
        self._session_id = session_id

    async def create(
        self, command: str, args: list[str], *, cwd: str,
    ) -> str:
        resp = await self._conn.create_terminal(
            command=command,
            args=list(args),
            cwd=cwd,
            session_id=self._session_id,
        )
        terminal_id = getattr(resp, "terminal_id", None)
        if not isinstance(terminal_id, str):
            raise RuntimeError("terminal/create returned no terminal_id")
        return terminal_id

    async def output(self, terminal_id: str) -> str:
        resp = await self._conn.terminal_output(
            session_id=self._session_id, terminal_id=terminal_id,
        )
        return str(getattr(resp, "output", "") or "")

    async def wait_for_exit(self, terminal_id: str) -> dict[str, Any]:
        resp = await self._conn.wait_for_terminal_exit(
            session_id=self._session_id, terminal_id=terminal_id,
        )
        # WaitForTerminalExitResponse carries an exit_status with exit_code
        # and/or signal. Flatten to a plain dict so callers don't depend on
        # the pydantic surface.
        exit_status = getattr(resp, "exit_status", None) or resp
        return {
            "exit_code": getattr(exit_status, "exit_code", None),
            "signal": getattr(exit_status, "signal", None),
        }

    async def kill(self, terminal_id: str) -> None:
        await self._conn.kill_terminal(
            session_id=self._session_id, terminal_id=terminal_id,
        )

    async def release(self, terminal_id: str) -> None:
        await self._conn.release_terminal(
            session_id=self._session_id, terminal_id=terminal_id,
        )
