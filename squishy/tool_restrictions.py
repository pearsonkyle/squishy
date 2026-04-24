"""Tool restrictions based on permission mode.

This module defines which tools are available in each permission mode
and provides utilities to check if a tool is allowed.
"""

from __future__ import annotations

import shlex
from typing import Any

READ_ONLY_TOOLS = frozenset({
    "read_file",
    "list_directory",
    "search_files",
    "glob_files",
    "recall",
    "plan_task",
    "update_plan",
    "get_plan",
    "save_note",
    "show_diff",
})

MUTATING_TOOLS = frozenset({
    "write_file",
    "edit_file",
})

SHELL_TOOLS = frozenset({
    "run_command",
})

ALL_TOOLS = READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOLS

# Plan tools excluded from bench mode to reduce schema size and prevent
# the model from wasting turns on planning instead of fixing.
_PLAN_TOOLS = frozenset({"plan_task", "update_plan", "get_plan"})

BENCH_TOOLS = ALL_TOOLS - _PLAN_TOOLS

# Shell commands allowed in plan mode. Single-word binaries are matched on the
# first token; two-word entries (e.g. "git log") match on the first two.
READONLY_SHELL_BINARIES = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find",
    "pwd", "which", "file", "stat", "tree", "env", "echo",
    "ruff", "mypy", "pyright",
})

READONLY_SHELL_TWO_WORD = frozenset({
    "git status", "git log", "git diff", "git show", "git branch",
    "git ls-files", "git blame",
    "pytest --collect-only",
})

READONLY_SHELL_THREE_WORD = frozenset({
    "python -m pytest",  # only paired with --collect-only (enforced below)
})

# Characters that can chain/redirect commands and escape the allowlist.
_SHELL_METACHARS = ("|", ";", "&", ">", "<", "`", "$(")


def is_readonly_shell(command: str) -> bool:
    """Return True if `command` is safe to run in plan mode.

    A command is safe when it has no shell metacharacters (which could chain
    in a mutating call) and its leading token (or two tokens, for things like
    `git log`) is in the allowlist.
    """
    if not isinstance(command, str):
        return False
    stripped = command.strip()
    if not stripped:
        return False
    if any(mc in stripped for mc in _SHELL_METACHARS):
        return False
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return False
    if not tokens:
        return False

    # Three-word match first (e.g. "python -m pytest --collect-only").
    if len(tokens) >= 3:
        three = f"{tokens[0]} {tokens[1]} {tokens[2]}"
        if three in READONLY_SHELL_THREE_WORD:
            if three == "python -m pytest" and "--collect-only" not in tokens[3:]:
                return False
            return True

    # Two-word match (e.g. "git log --oneline").
    if len(tokens) >= 2:
        two = f"{tokens[0]} {tokens[1]}"
        if two in READONLY_SHELL_TWO_WORD:
            return True

    return tokens[0] in READONLY_SHELL_BINARIES


def get_allowed_tools(mode: str) -> frozenset[str]:
    """Return set of tool names allowed in given mode.

    In `plan` mode, `run_command` is conditionally allowed (gated per-call by
    `is_readonly_shell`). It is included here so schemas can expose it to the
    model; `check_permission` rejects unsafe invocations at dispatch time.
    """
    if mode == "yolo":
        return ALL_TOOLS
    if mode == "bench":
        return BENCH_TOOLS
    if mode == "edits":
        return READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOLS
    if mode == "plan":
        return READ_ONLY_TOOLS | SHELL_TOOLS
    return frozenset()


def check_permission(
    tool_name: str,
    mode: str,
    args: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (allowed, reason).

    `reason == "prompt"` means the caller should ask the user before executing.
    In plan mode, `run_command` is permitted only when its `command` is on the
    read-only allowlist.
    """
    allowed = get_allowed_tools(mode)

    if tool_name not in allowed:
        if mode == "plan":
            return False, "refused: plan mode is read-only"
        if mode == "edits" and tool_name in SHELL_TOOLS:
            return False, "prompt"
        return False, f"refused: tool {tool_name} not available in {mode} mode"

    # Plan-mode gating for run_command.
    if mode == "plan" and tool_name == "run_command":
        cmd = (args or {}).get("command", "") if isinstance(args, dict) else ""
        if not is_readonly_shell(cmd if isinstance(cmd, str) else ""):
            return False, (
                "refused: plan mode allows only read-only shell commands "
                "(ls, cat, head, tail, grep, rg, find, wc, pwd, which, file, "
                "stat, tree, ruff check, mypy, pyright, "
                "git status/log/diff/show/branch/blame/ls-files, "
                "pytest --collect-only, python -m pytest --collect-only). "
                "No pipes, redirects, or command chains."
            )

    # Edits-mode shell prompt (only reached when tool IS in allowed set; for
    # run_command we also want the approval prompt to still fire).
    if mode == "edits" and tool_name in SHELL_TOOLS:
        return False, "prompt"

    return True, ""


