"""Tool restrictions based on permission mode.

This module defines which tools are available in each permission mode
and provides utilities to check if a tool is allowed.
"""

from __future__ import annotations

import re
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
    "finish_plan",
    "save_note",
    "show_diff",
})

MUTATING_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "undo_edit",
})

WEB_TOOLS = frozenset({
    "fetch_url",
})

SHELL_TOOL_NAMES = frozenset({
    "run_command",
})

ALL_TOOLS = READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOL_NAMES | WEB_TOOLS

# Bench mode is air-gapped — `fetch_url` cannot work and just bloats the
# tool schema (~150 tokens × every bench call).  Drop it.
BENCH_TOOLS = ALL_TOOLS - WEB_TOOLS

# Shell commands allowed in plan mode. Single-word binaries are matched on the
# first token; two-word entries (e.g. "git log") match on the first two.
READONLY_SHELL_BINARIES = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find",
    "pwd", "which", "file", "stat", "tree", "printenv", "echo",
    "ruff", "mypy", "pyright",
    # `cd` writes nothing to disk. Allowing it as a segment lets the agent
    # prefix chains like `cd subdir && ruff check .` without bouncing off
    # the allowlist — every other segment is still checked independently.
    "cd",
})

READONLY_SHELL_TWO_WORD = frozenset({
    "git status", "git log", "git diff", "git show", "git branch",
    "git ls-files", "git blame",
    "pytest --collect-only",
})

READONLY_SHELL_THREE_WORD = frozenset({
    "python -m pytest",  # only paired with --collect-only (enforced below)
})

# Hard-rejected anywhere in a plan-mode command. These all enable
# command substitution, env-var expansion, or stray newlines that
# break our token analysis.
_HARD_REJECT = ("`", "$(", "${", "\n", "\r")
# File redirection — disallowed in plan mode because `> file` writes
# to disk. Stderr fd swaps like `2>&1` are stripped before this check.
_FILE_REDIRECT = ("<", ">")
# Pure fd redirections that don't touch the filesystem (e.g. `2>&1`,
# `1>&2`, `>&2`). Safe in plan mode — we strip them before tokenising.
_FD_REDIRECT_RE = re.compile(r"\s*\d*>&\d+")
# Operators that chain multiple commands. We split on these and require
# every segment to independently be a read-only command.
_CHAIN_RE = re.compile(r"\|\||&&|;|\|")


def is_readonly_shell(command: str) -> bool:
    """Return True if ``command`` is safe to run in plan mode.

    Plan mode allows:
      * a single command whose leading token(s) are in the allowlist;
      * stderr→stdout fd redirection (``2>&1``, ``>&2``) — fd swap only;
      * pipes (``|``) and chains (``;``, ``&&``, ``||``) where *every*
        segment is independently allowlisted.

    Plan mode rejects:
      * file redirection (``>``, ``<``) — would write or read arbitrary
        paths;
      * command substitution (`` ` ``, ``$(``, ``${``) — runs other
        commands or expands variables we can't reason about;
      * background jobs (``&``) and stray newlines.
    """
    if not isinstance(command, str):
        return False
    stripped = command.strip()
    if not stripped:
        return False

    # Strip pure fd redirections before any other check so `ruff check . 2>&1`
    # tokenises as just `ruff check .`.
    stripped = _FD_REDIRECT_RE.sub("", stripped).strip()
    if not stripped:
        return False

    if any(mc in stripped for mc in _HARD_REJECT):
        return False
    if any(mc in stripped for mc in _FILE_REDIRECT):
        return False

    segments = [seg.strip() for seg in _CHAIN_RE.split(stripped) if seg.strip()]
    if not segments:
        return False
    for seg in segments:
        # Lone `&` (background) survives the chain split — reject it.
        if "&" in seg:
            return False
        if not _segment_is_readonly(seg):
            return False
    return True


def _segment_is_readonly(segment: str) -> bool:
    """Check that a single command segment matches the readonly allowlist."""
    try:
        tokens = shlex.split(segment)
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
        return READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOL_NAMES
    if mode == "plan":
        return READ_ONLY_TOOLS | SHELL_TOOL_NAMES
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
    # MCP tools: apply mode-based filtering.
    if tool_name.startswith("mcp__"):
        if mode == "plan":
            return False, "refused: MCP tools are blocked in plan mode (switch to edits or yolo)"
        if mode == "edits":
            return False, "prompt"
        return True, ""

    allowed = get_allowed_tools(mode)

    if tool_name not in allowed:
        if mode == "plan":
            return False, "refused: plan mode is read-only"
        return False, f"refused: tool {tool_name} not available in {mode} mode"

    # Plan-mode gating for run_command.
    if mode == "plan" and tool_name == "run_command":
        cmd = (args or {}).get("command", "") if isinstance(args, dict) else ""
        if not is_readonly_shell(cmd if isinstance(cmd, str) else ""):
            return False, (
                "refused: plan mode is read-only. Allowed binaries: "
                "ls, cat, head, tail, wc, grep, rg, find, pwd, which, file, "
                "stat, tree, printenv, echo, cd, ruff, mypy, pyright; "
                "git status/log/diff/show/branch/blame/ls-files; "
                "pytest --collect-only; python -m pytest --collect-only. "
                "Pipes/chains and `2>&1` are fine if every segment is allowed. "
                "Rejected: file redirects (`>`, `<`), command substitution "
                "(`` ` ``, `$(...)`, `${...}`), background jobs (`&`), and "
                "arbitrary scripts like `python -c`. "
                "Note: `run_command` already runs in the project root — no `cd` needed. "
                "For inspection, prefer the dedicated tools "
                "(`list_directory`, `read_file`, `search_files`, `glob_files`)."
            )

    # Edits-mode shell prompt (only reached when tool IS in allowed set; for
    # run_command we also want the approval prompt to still fire).
    if mode == "edits" and tool_name in SHELL_TOOL_NAMES:
        return False, "prompt"

    return True, ""


