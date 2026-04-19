"""Tool restrictions based on permission mode.

This module defines which tools are available in each permission mode
and provides utilities to check if a tool is allowed.
"""

from __future__ import annotations

READ_ONLY_TOOLS = frozenset({
    "read_file",
    "list_directory",
    "search_files",
    "recall",
})

MUTATING_TOOLS = frozenset({
    "write_file",
    "edit_file",
})

SHELL_TOOLS = frozenset({
    "run_command",
})

# Tools that manage the plan itself. `exit_plan_mode` only makes sense in plan
# mode. `update_plan_step` is always allowed so the model can mark progress
# after the plan is approved (session has moved to edits mode).
PLAN_MODE_ONLY = frozenset({"exit_plan_mode"})
PLANNING_TOOLS = PLAN_MODE_ONLY | frozenset({"update_plan_step"})

ALL_TOOLS = READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOLS | PLANNING_TOOLS


def is_read_only(tool_name: str) -> bool:
    """Check if a tool is read-only (safe in plan mode)."""
    return tool_name in READ_ONLY_TOOLS


def is_mutating(tool_name: str) -> bool:
    """Check if a tool mutates the filesystem."""
    return tool_name in MUTATING_TOOLS


def is_shell(tool_name: str) -> bool:
    """Check if a tool executes shell commands."""
    return tool_name in SHELL_TOOLS


def get_allowed_tools(mode: str) -> frozenset[str]:
    """Return set of tool names allowed in given mode."""
    # update_plan_step is always available so the model can mark progress in
    # any mode. exit_plan_mode is plan-mode-only.
    always = frozenset({"update_plan_step"})
    if mode == "yolo":
        return ALL_TOOLS
    if mode == "edits":
        return READ_ONLY_TOOLS | MUTATING_TOOLS | always
    if mode == "plan":
        return READ_ONLY_TOOLS | PLANNING_TOOLS
    return frozenset()


def get_denied_tools(mode: str) -> frozenset[str]:
    """Return set of tool names denied in given mode."""
    return ALL_TOOLS - get_allowed_tools(mode)


def check_permission(tool_name: str, mode: str) -> tuple[bool, str]:
    """Return (allowed, reason).

    ``reason == "prompt"`` means the caller should ask the user before
    executing. Any other non-empty reason is a hard refusal.
    """
    allowed = get_allowed_tools(mode)

    if tool_name in allowed:
        return True, ""

    if tool_name in PLAN_MODE_ONLY:
        return False, f"refused: {tool_name} is only available in plan mode"
    if mode == "plan":
        return False, "refused: plan mode is read-only"
    if mode == "edits" and tool_name in SHELL_TOOLS:
        return False, "prompt"
    return False, f"refused: {tool_name} not allowed in {mode} mode"


def get_tool_category(tool_name: str) -> str:
    """Return category of tool for display purposes."""
    if is_read_only(tool_name):
        return "read-only"
    if is_mutating(tool_name):
        return "mutating"
    if is_shell(tool_name):
        return "shell"
    return "unknown"
