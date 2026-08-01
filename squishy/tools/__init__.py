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
from squishy.tool_restrictions import get_profile_tools as _get_profile_tools
from squishy.tools import pressure
from squishy.tools.base import Tool, ToolContext, ToolResult
from squishy.tools.fs import FS_TOOLS
from squishy.tools.graph import GRAPH_TOOLS
from squishy.tools.recall import RECALL_TOOLS
from squishy.tools.scratchpad import SCRATCHPAD_TOOLS
from squishy.tools.shell import SHELL_TOOLS
from squishy.tools.web import WEB_TOOLS

ALL_TOOLS: list[Tool] = [
    *FS_TOOLS, *RECALL_TOOLS, *GRAPH_TOOLS, *SHELL_TOOLS, *SCRATCHPAD_TOOLS, *WEB_TOOLS
]
REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}
_GRAPH_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in GRAPH_TOOLS)

# PromptFn returns either a bool (approve/decline) or a ("feedback", str) tuple
# where the string is free-text feedback the agent should use to revise.
PromptFn = Callable[[Tool, dict[str, object]], Awaitable[Any]]


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
            reply = await prompt_fn(tool, args)
            if reply is not True:
                return ToolResult(False, error="refused: user declined")
        else:
            return ToolResult(False, error=reason)

    try:
        result = await tool.run(args, ctx)
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, error=f"{type(e).__name__}: {e}")
    # Edit pressure rides on the result of the call that earned it. Applied
    # here rather than in each tool so every tool carries it and no tool has
    # to remember to.
    return pressure.apply(ctx, tool.name, result)


def openai_schemas(
    mode: str | None = None,
    *,
    profile: str = "standard",
    extra_tools: frozenset[str] | set[str] | None = None,
    has_index: bool = True,
    has_graph: bool = True,
) -> list[dict[str, object]]:
    """Return OpenAI-format tool schemas, optionally filtered by mode.

    When ``mode`` is None, all tools are returned (backwards compatibility).
    When ``mode`` is set, only tools permitted in that mode are exposed.

    ``profile`` narrows the result further (see ``tool_restrictions``):
    ``"minimal"`` exposes only a shell plus the file primitives, which is the
    shape most small models have actually been trained on; ``"shell"`` exposes
    the shell alone. ``extra_tools`` adds names back on top of a profile — used
    to surface ``recall`` only when an index exists, rather than advertising a
    tool that would immediately error. A profile only shapes the *schema*; it
    adds no refusal path, so a model that calls an unlisted tool from memory is
    still served.

    The schema is stable for the whole run. It used to be recomputed per turn
    so gates could withdraw tools mid-run; that never worked — a model with a
    dozen turns of ``run_command`` in its history keeps calling it from history
    regardless of what the schema says — and the withdrawals repeatedly left
    narrow profiles with no callable tool at all.
    """
    if mode is None:
        return [t.openai_schema() for t in ALL_TOOLS]

    narrow = _get_profile_tools(profile)
    if narrow is not None and extra_tools:
        narrow = narrow | frozenset(extra_tools)

    def _keep(name: str) -> bool:
        # `recall` without an index can only ever answer "no index found. Run
        # /init first" — a wasted call, and one models kept making. Hide it
        # instead of advertising a tool that cannot work here.
        if name == "recall" and not has_index:
            return False
        # Same rule for the graph tools: with no .squishy/graph.json they can
        # only answer "run /init first", which is a turn spent to learn what
        # the schema could have said for free.
        if name in _GRAPH_TOOL_NAMES and not has_graph:
            return False
        return narrow is None or name in narrow

    allowed = _get_allowed_tools(mode)
    return [
        t.openai_schema()
        for t in ALL_TOOLS
        if (t.name in allowed or t.name.startswith("mcp__")) and _keep(t.name)
    ]


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
