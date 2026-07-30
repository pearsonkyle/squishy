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
from squishy.tools.base import Tool, ToolContext, ToolResult
from squishy.tools.fs import FS_TOOLS
from squishy.tools.plan import PLAN_TOOLS
from squishy.tools.recall import RECALL_TOOLS
from squishy.tools.scratchpad import SCRATCHPAD_TOOLS
from squishy.tools.shell import SHELL_TOOLS
from squishy.tools.web import WEB_TOOLS

ALL_TOOLS: list[Tool] = [*FS_TOOLS, *RECALL_TOOLS, *SHELL_TOOLS, *PLAN_TOOLS, *SCRATCHPAD_TOOLS, *WEB_TOOLS]
REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}

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

    blocked = ctx.blocked_tools.get(name)
    if blocked:
        return ToolResult(False, error=blocked)

    allowed, reason = check_permission(tool, ctx.permission_mode, args)
    if not allowed:
        if reason == "prompt":
            if prompt_fn is None:
                return ToolResult(False, error="refused: user approval required (no TTY)")
            reply = await prompt_fn(tool, args)
            # prompt_fn may return a ("feedback", text) tuple for plan_task;
            # for any non-plan tool we just treat that as a decline.
            if reply is not True:
                return ToolResult(False, error="refused: user declined")
        else:
            return ToolResult(False, error=reason)

    try:
        return await tool.run(args, ctx)
    except Exception as e:  # noqa: BLE001
        return ToolResult(False, error=f"{type(e).__name__}: {e}")


def openai_schemas(
    mode: str | None = None,
    *,
    plan_active: bool = False,
    phase: str | None = None,
    profile: str = "standard",
    extra_tools: frozenset[str] | set[str] | None = None,
) -> list[dict[str, object]]:
    """Return OpenAI-format tool schemas, optionally filtered by mode and phase.

    When ``mode`` is None, all tools are returned (backwards compatibility).
    When ``mode`` is set, only tools permitted in that mode are exposed — so
    the model never sees ``write_file``/``edit_file`` in plan mode, etc.

    When ``phase`` is set and ``mode`` is ``"bench"``, tools are further
    filtered to only those available in that phase.  This is the primary
    mechanism for phase-gated behaviour — the model literally cannot call
    tools that are not in its schema for the current phase.

    ``plan_active`` should be True once a plan_task has been approved.
    The schema then hides ``plan_task`` so the model can't restart
    planning instead of executing — it must use ``update_plan`` /
    ``finish_plan`` instead. ``update_plan`` itself supports
    ``add_steps`` for genuine scope changes.

    ``profile`` narrows the result further (see ``tool_restrictions``):
    ``"minimal"`` exposes only a shell plus the file primitives, which is the
    shape most small models have actually been trained on. ``extra_tools``
    adds names back on top of a profile — used to surface ``recall`` only when
    an index exists, rather than advertising a tool that would immediately
    error. A profile only shapes the *schema*; it adds no refusal path, so a
    model that calls an unlisted tool from memory is still served.
    """
    if mode is None:
        return [t.openai_schema() for t in ALL_TOOLS]

    narrow = _get_profile_tools(profile)
    if narrow is not None and extra_tools:
        narrow = narrow | frozenset(extra_tools)

    def _keep(name: str) -> bool:
        return narrow is None or name in narrow

    # Phase-gated filtering for bench mode.
    if phase is not None and mode == "bench":
        from squishy.phase_machine import tools_for_phase
        phase_tools = tools_for_phase(phase)
        return [
            t.openai_schema()
            for t in ALL_TOOLS
            if t.name in phase_tools
            and _keep(t.name)
            and not (plan_active and t.name == "plan_task")
        ]

    # Standard mode-based filtering (all other modes, or bench without phase).
    allowed = _get_allowed_tools(mode)
    return [
        t.openai_schema()
        for t in ALL_TOOLS
        if (t.name in allowed or (t.name.startswith("mcp__") and mode != "plan"))
        and _keep(t.name)
        and not (plan_active and t.name == "plan_task")
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
