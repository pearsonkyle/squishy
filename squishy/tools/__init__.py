"""Tool registry + async dispatch.

Single source of truth for what the model can call:
- rendered into OpenAI `tools=[...]` schemas (client.py)
- looked up by name at dispatch time (agent.py)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from squishy.tool_restrictions import check_permission as _check_permission
from squishy.tool_restrictions import get_allowed_tools as _get_allowed_tools
from squishy.tools.base import Tool, ToolContext, ToolResult
from squishy.tools.fs import FS_TOOLS
from squishy.tools.plan import PLAN_TOOLS
from squishy.tools.recall import RECALL_TOOLS
from squishy.tools.scratchpad import SCRATCHPAD_TOOLS
from squishy.tools.shell import SHELL_TOOLS

ALL_TOOLS: list[Tool] = [*FS_TOOLS, *RECALL_TOOLS, *SHELL_TOOLS, *PLAN_TOOLS, *SCRATCHPAD_TOOLS]
REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}

PromptFn = Callable[[Tool, dict[str, object]], Awaitable[bool]]


def check_permission(
    tool: Tool,
    mode: str,
    args: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (allowed, reason).

    reason == "prompt" means the caller should ask the user before executing.
    """
    return _check_permission(tool.name, mode, args)


async def dispatch(
    name: str,
    args: dict[str, object],
    ctx: ToolContext,
    prompt_fn: PromptFn | None = None,
) -> ToolResult:
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult(False, error=f"unknown tool: {name}")

    # Surface JSON-argument parse errors from the client before we hand the
    # (possibly malformed) args to the tool. The tool would otherwise return
    # a misleading "missing required field" error.
    tool_arg_error = args.get("_tool_arg_error")
    if isinstance(tool_arg_error, str):
        return ToolResult(False, error=tool_arg_error)

    allowed, reason = check_permission(tool, ctx.permission_mode, args)
    if not allowed:
        if reason == "prompt":
            if prompt_fn is None:
                return ToolResult(False, error="refused: user approval required (no TTY)")
            if not await prompt_fn(tool, args):
                return ToolResult(False, error="refused: user declined")
        else:
            return ToolResult(False, error=reason)

    try:
        return await tool.run(args, ctx)
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, error=f"{type(e).__name__}: {e}")


def openai_schemas(mode: str | None = None) -> list[dict[str, object]]:
    """Return OpenAI-format tool schemas, optionally filtered by mode.

    When `mode` is None, all tools are returned (backwards compatibility).
    When `mode` is set, only tools permitted in that mode are exposed — so the
    model never sees `write_file`/`edit_file` in plan mode, etc.
    """
    if mode is None:
        return [t.openai_schema() for t in ALL_TOOLS]
    allowed = _get_allowed_tools(mode)
    return [t.openai_schema() for t in ALL_TOOLS if t.name in allowed]


__all__ = [
    "ALL_TOOLS",
    "REGISTRY",
    "PromptFn",
    "Tool",
    "ToolContext",
    "ToolResult",
    "check_permission",
    "dispatch",
    "openai_schemas",
]
