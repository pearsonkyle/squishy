"""Adapter that turns the ACP ``Client`` filesystem methods into the simple
``read_text_file`` / ``write_text_file`` shape squishy's fs tools expect.

When an ACP client advertises ``fs.readTextFile`` / ``fs.writeTextFile``, the
agent routes file IO through it so the editor's diff view stays in sync with
the agent's changes and unsaved buffers can override on-disk content.
"""
from __future__ import annotations

from typing import Any


class AcpFsClient:
    """Thin wrapper around the agent-side ACP connection.

    Only used when the client capability bit is set. Each call binds the
    current session id and translates back to plain strings so the fs tools
    don't have to know anything about ACP.
    """

    def __init__(self, conn: Any, session_id: str, *, read: bool, write: bool) -> None:
        self._conn = conn
        self._session_id = session_id
        self._can_read = read
        self._can_write = write

    @property
    def can_read(self) -> bool:
        return self._can_read

    @property
    def can_write(self) -> bool:
        return self._can_write

    async def read_text_file(self, path: str) -> str:
        if not self._can_read:
            raise OSError("ACP client did not advertise fs.readTextFile")
        resp = await self._conn.read_text_file(path=path, session_id=self._session_id)
        # The acp schema response object exposes the content as `.content`.
        return str(getattr(resp, "content", "") or "")

    async def write_text_file(self, path: str, content: str) -> None:
        if not self._can_write:
            raise OSError("ACP client did not advertise fs.writeTextFile")
        await self._conn.write_text_file(
            path=path, content=content, session_id=self._session_id,
        )
