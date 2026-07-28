"""`save_note` tool — persist findings across conversation history trims.

Notes are stored on ToolContext.notes and injected as a synthetic system
message each turn, surviving history trimming the same way plan-status
messages do.
"""

from __future__ import annotations

from typing import Any

from squishy.tools.base import Tool, ToolContext, ToolResult

MAX_NOTES = 10
MAX_NOTE_CHARS = 2000
NOTES_TAG = "<notes>"
NOTES_END_TAG = "</notes>"


def is_notes_message(msg: dict[str, Any]) -> bool:
    """Return True if this message is a notes-injection system message."""
    return (
        msg.get("role") == "system"
        and isinstance(msg.get("content"), str)
        and NOTES_TAG in msg["content"]
    )


def render_notes(notes: dict[str, str]) -> str:
    """Render all notes into a system message block."""
    if not notes:
        return ""
    lines = [f"- **{key}**: {value}" for key, value in notes.items()]
    return f"{NOTES_TAG}\n## Saved Notes\n" + "\n".join(lines) + f"\n{NOTES_END_TAG}"


async def _save_note(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = args.get("key")
    content = args.get("content")

    if not isinstance(key, str) or not key.strip():
        return ToolResult(False, error="`key` is required (string)")
    if not isinstance(content, str) or not content.strip():
        return ToolResult(False, error="`content` is required (string)")

    key = key.strip()
    content = content.strip()

    # Cap content length
    if len(content) > MAX_NOTE_CHARS:
        content = content[:MAX_NOTE_CHARS] + "…(truncated)"

    # Evict oldest if at capacity (and this is a new key)
    evicted_key = None
    if key not in ctx.notes and len(ctx.notes) >= MAX_NOTES:
        evicted_key = next(iter(ctx.notes))
        del ctx.notes[evicted_key]

    ctx.notes[key] = content
    data: dict[str, Any] = {"key": key, "saved": True, "total_notes": len(ctx.notes)}
    display = f"saved note '{key}' ({len(content)} chars, {len(ctx.notes)} total)"
    if evicted_key:
        data["evicted"] = evicted_key
        data["warning"] = f"Note '{evicted_key}' was evicted to make room (max {MAX_NOTES} notes)"
        display += f" — evicted oldest note '{evicted_key}'"
    return ToolResult(True, data=data, display=display)


save_note = Tool(
    name="save_note",
    description=(
        "Persist a short note (key + content) that survives context trimming. "
        "Use for findings, file paths, or decisions to remember across turns."
    ),
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short label, e.g. 'root_cause'"},
            "content": {"type": "string"},
        },
        "required": ["key", "content"],
    },
    run=_save_note,
)

SCRATCHPAD_TOOLS: list[Tool] = [save_note]

__all__ = ["save_note", "SCRATCHPAD_TOOLS", "is_notes_message", "render_notes"]
