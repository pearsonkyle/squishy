"""Tool registry + async dispatch.
 
Single source of truth for what the model can call:
- rendered into OpenAI `tools=[...]` schemas (client.py)
- looked up by name at dispatch time (agent.py)
"""
 
from __future__ import annotations
 
from collections.abc import Awaitable, Callable
 
from squishy.tools.base import Tool, ToolContext, ToolResult
from squishy.tools.fs import FS_TOOLS
from squishy.tools.recall import RECALL_TOOLS
from squishy.tools.shell import SHELL_TOOLS
 
ALL_TOOLS: list[Tool] = [*FS_TOOLS, *RECALL_TOOLS, *SHELL_TOOLS]
REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}
 
PromptFn = Callable[[Tool, dict[str, object]], Awaitable[bool]]
 
 
def check_permission(tool: Tool, mode: str) -> tuple[bool, str]:
    """Return (allowed, reason).
 
    reason == "prompt" means the caller should ask the user before executing.
    """
    if mode == "yolo":
        return True, ""
    if not tool.mutates:
        return True, ""
    if mode == "edits":
        if tool.name == "run_command":
            return False, "prompt"
        return True, ""
    if mode == "plan":
        return False, "refused: plan mode is read-only"
    return False, f"unknown permission mode: {mode}"
 
 
async def dispatch(
    name: str,
    args: dict[str, object],
    ctx: ToolContext,
    prompt_fn: PromptFn | None = None,
) -> ToolResult:
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult(False, error=f"unknown tool: {name}")
 
    allowed, reason = check_permission(tool, ctx.permission_mode)
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
 
 
def openai_schemas() -> list[dict[str, object]]:
    return [t.openai_schema() for t in ALL_TOOLS]
 
 
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