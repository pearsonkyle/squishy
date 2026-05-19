"""``Display``-shaped adapter that forwards every UI event as an ACP
``session/update`` notification.

The squishy Agent loop calls a ``Display`` (see ``squishy/display.py``) for
streaming text, tool announcements, edit diffs, command output, and plan
progress. The terminal Display renders these to a TTY; the ACP Display
converts the same events into editor-facing ``SessionNotification`` updates
so the editor's chat pane, tool-call cards, diff view, and plan panel all
update live.

The Display interface is synchronous, but ACP notifications are async. To
bridge that, the adapter schedules fire-and-forget tasks on the event loop
and exposes ``flush()`` so the ACP session handler can await them before
sending the final ``PromptResponse``.

Pattern mirrors ``squishy.api._CallbackDisplay`` (squishy/api.py:197) — a
``__getattr__`` fallback absorbs any Display method we don't care about
without breaking the agent loop.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from acp import helpers
from acp.schema import AvailableCommand, ToolCallProgress, ToolCallStart

from squishy.display import Stats

log = logging.getLogger("squishy.acp.display")

# Map squishy tool names to ACP tool kinds for richer editor cards.
_TOOL_KIND: dict[str, str] = {
    "read_file": "read",
    "list_directory": "read",
    "search_files": "search",
    "glob_files": "search",
    "recall": "search",
    "write_file": "edit",
    "edit_file": "edit",
    "undo_edit": "edit",
    "run_command": "execute",
    "fetch_url": "fetch",
    "plan_task": "think",
    "update_plan": "think",
    "finish_plan": "think",
    "save_note": "other",
    "show_diff": "read",
    "get_plan": "read",
}


class AcpDisplay:
    """Bridge a squishy ``Display`` to ACP ``session/update`` notifications.

    One instance per session. Constructed with the agent-side connection
    (``acp.Client``) and the active session id, plus the asyncio loop the
    Agent runs on so the sync Display callbacks can schedule async sends.
    """

    def __init__(self, conn: Any, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._conn = conn
        self._session_id = session_id
        self._loop = loop
        self._pending: list[asyncio.Task[Any]] = []
        # Track the most recent tool call so we can correlate the result
        # (which arrives via a separate Display call) with the right id.
        self._current_tool_id: str | None = None
        self._current_tool_name: str | None = None
        # squishy code path expects ``display.stats`` and ``display.console``.
        self.stats = Stats()
        self.console = _NullConsole()
        self.model: str = ""
        self.mode: str = ""

    # ── async plumbing ────────────────────────────────────────────────

    def _emit(self, update: Any) -> None:
        """Schedule an async session_update on the agent's event loop."""
        if self._conn is None:
            return
        try:
            task = self._loop.create_task(
                self._conn.session_update(session_id=self._session_id, update=update),
            )
        except RuntimeError:
            # Loop has been closed — happens during shutdown. Drop the update.
            return
        self._pending.append(task)
        # Best-effort: prune already-done tasks so the list doesn't grow.
        self._pending = [t for t in self._pending if not t.done()]

    async def flush(self) -> None:
        """Await every queued notification before returning to the client."""
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        results = await asyncio.gather(*pending, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.warning("session_update failed: %s", r)

    # ── streaming text ────────────────────────────────────────────────

    def streaming_text_chunk(self, s: str) -> None:
        if not s:
            return
        self._emit(helpers.update_agent_message_text(s))

    def flush_streaming_text(self) -> None:  # no-op — chunks were emitted live
        return

    def reset_streaming(self) -> None:
        return

    def start_thinking(self, label: str = "thinking") -> None:
        return

    def stop_thinking(self) -> None:
        return

    # ── tool call lifecycle ───────────────────────────────────────────

    def turn_header(
        self,
        turn: int,
        max_turns: int,
        tool_name: str,
        brief: str,
        mode: str | None = None,
    ) -> None:
        tool_call_id = uuid.uuid4().hex[:16]
        self._current_tool_id = tool_call_id
        self._current_tool_name = tool_name
        update: ToolCallStart = helpers.start_tool_call(
            tool_call_id=tool_call_id,
            title=f"{tool_name}{(' ' + brief) if brief else ''}",
            kind=_TOOL_KIND.get(tool_name, "other"),
            status="in_progress",
        )
        self._emit(update)

    def tool_result(self, success: bool, display: str, duration_ms: float) -> None:
        tool_call_id = self._current_tool_id
        if tool_call_id is None:
            return
        status = "completed" if success else "failed"
        content = None
        if display:
            content = [helpers.tool_content(helpers.text_block(display))]
        update: ToolCallProgress = helpers.update_tool_call(
            tool_call_id=tool_call_id,
            status=status,
            content=content,
        )
        self._emit(update)

    def command_line(self, command: str) -> None:
        if self._current_tool_id is None:
            return
        self._emit(helpers.update_tool_call(
            tool_call_id=self._current_tool_id,
            content=[helpers.tool_content(helpers.text_block(f"$ {command}"))],
        ))

    def command_output(self, data: dict[str, Any]) -> None:
        if self._current_tool_id is None:
            return
        parts: list[str] = []
        stdout = str(data.get("stdout", "")).rstrip()
        stderr = str(data.get("stderr", "")).rstrip()
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append("[stderr]\n" + stderr)
        if not parts:
            return
        self._emit(helpers.update_tool_call(
            tool_call_id=self._current_tool_id,
            content=[helpers.tool_content(helpers.text_block("\n".join(parts)))],
        ))

    def edit_diff(self, path: str, old: str, new: str) -> None:
        if self._current_tool_id is None:
            return
        self._emit(helpers.update_tool_call(
            tool_call_id=self._current_tool_id,
            content=[helpers.tool_diff_content(path=path, new_text=new, old_text=old)],
        ))

    def write_preview(self, path: str, content: str) -> None:
        if self._current_tool_id is None:
            return
        self._emit(helpers.update_tool_call(
            tool_call_id=self._current_tool_id,
            content=[helpers.tool_diff_content(path=path, new_text=content, old_text=None)],
        ))

    # ── plan progress ────────────────────────────────────────────────

    def plan_panel(self, data: dict[str, Any]) -> None:
        entries = _plan_entries_from_dict(data)
        if entries:
            self._emit(helpers.update_plan(entries))

    def plan_progress(self, steps: list[dict[str, Any]]) -> None:
        entries = [_plan_entry_from_step(s) for s in steps if isinstance(s, dict)]
        if entries:
            self._emit(helpers.update_plan(entries))

    # ── informational messages ────────────────────────────────────────

    def info(self, s: str) -> None:
        if s:
            self._emit(helpers.update_agent_message_text(f"{s}\n"))

    def warn(self, s: str) -> None:
        self.info(s)

    def error(self, s: str) -> None:
        self.info(s)

    def nudge(self, content: str) -> None:
        self.info(content)

    def mode_changed(self, mode: str) -> None:
        self.mode = mode
        self._emit(helpers.update_current_mode(mode))

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    # ── available commands ───────────────────────────────────────────

    def emit_available_commands(self, tool_names: list[str]) -> None:
        cmds = [
            AvailableCommand(name=name, description=_tool_description(name))
            for name in tool_names
        ]
        self._emit(helpers.update_available_commands(cmds))

    # ── unused Display methods absorbed silently ─────────────────────

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


class _NullConsole:
    """Stand-in for a rich Console — squishy.display sometimes accesses
    ``.console.print``/``.console.out`` directly from helpers we don't want
    to refactor."""

    def print(self, *args: Any, **kwargs: Any) -> None:
        return

    def out(self, *args: Any, **kwargs: Any) -> None:
        return


def _plan_entry_from_step(step: dict[str, Any]) -> Any:
    from squishy.plan_state import _ACP_STATUS_MAP  # local import to avoid cycles

    desc = str(step.get("description", "")) or ""
    status = _ACP_STATUS_MAP.get(str(step.get("status", "pending")), "pending")
    note = str(step.get("note", "")) if step.get("status") == "blocked" else ""
    if note:
        desc = f"{desc} (blocked: {note})"
    return helpers.plan_entry(desc, status=status, priority="medium")


def _plan_entries_from_dict(plan_dict: dict[str, Any]) -> list[Any]:
    steps = plan_dict.get("steps")
    if not isinstance(steps, list):
        return []
    return [_plan_entry_from_step(s) for s in steps if isinstance(s, dict)]


def _tool_description(name: str) -> str:
    """Render a one-line description for an AvailableCommand entry."""
    from squishy.tools import REGISTRY

    tool = REGISTRY.get(name)
    if tool is None:
        return name
    desc = tool.description or name
    # AvailableCommand.description should be a short sentence.
    return desc.splitlines()[0][:160]
