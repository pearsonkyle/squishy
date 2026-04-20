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
    "plan_task",
    "update_plan",
})

MUTATING_TOOLS = frozenset({
    "write_file",
    "edit_file",
})

SHELL_TOOLS = frozenset({
    "run_command",
})

ALL_TOOLS = READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOLS


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
    if mode == "yolo":
        return ALL_TOOLS
    if mode == "edits":
        return READ_ONLY_TOOLS | MUTATING_TOOLS
    if mode == "plan":
        return READ_ONLY_TOOLS
    return frozenset()


def get_denied_tools(mode: str) -> frozenset[str]:
    """Return set of tool names denied in given mode."""
    return ALL_TOOLS - get_allowed_tools(mode)


def check_permission(tool_name: str, mode: str) -> tuple[bool, str]:
    """Return (allowed, reason)."""
    allowed = get_allowed_tools(mode)
    
    if tool_name not in allowed:
        if mode == "plan":
            return False, "refused: plan mode is read-only"
        if mode == "edits" and tool_name in SHELL_TOOLS:
            return False, "prompt"
        if mode == "yolo":
            pass  # All tools allowed
    
    return True, ""


def get_tool_category(tool_name: str) -> str:
    """Return category of tool for display purposes."""
    if is_read_only(tool_name):
        return "read-only"
    if is_mutating(tool_name):
        return "mutating"
    if is_shell(tool_name):
        return "shell"
    return "unknown"
