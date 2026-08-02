"""Tool restrictions based on permission mode.

This module defines which tools are available in each permission mode
and provides utilities to check if a tool is allowed.
"""

from __future__ import annotations

from typing import Any

READ_ONLY_TOOLS = frozenset({
    "read_file",
    "list_directory",
    "search_files",
    "glob_files",
    "recall",
    "explore",
    "impact_of",
    "repo_map",
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

# --- Tool profiles --------------------------------------------------------
#
# A profile narrows what the model *sees* in its schema; it never widens what
# permission mode allows, and it does not add a refusal path. If a model calls
# something outside its profile (many are trained on Claude Code / mini-swe /
# pi tool vocabularies), the call still dispatches normally — `tool_aliases`
# already maps the common alternate spellings onto our canonical names.
#
# `minimal` is the mini-swe-agent shape: a shell plus the file primitives a
# small model actually needs. Everything else (planning, scratchpad, diffing,
# directory listing, globbing, grepping) is reachable through `run_command`,
# so exposing dedicated tools for them buys little and costs schema tokens on
# every single request.
MINIMAL_TOOLS = frozenset({
    "run_command", "read_file", "edit_file", "write_file",
})

# `minimal` plus the one graph tool that replaces a crawl. `explore` earns its
# ~120 schema tokens by answering in one call what grep-then-read-then-read
# answers in four; `impact_of` and `repo_map` do not, so they stay out — a
# narrow profile exists to be narrow. Only offered when a graph exists.
GRAPH_TOOLS = frozenset({
    "run_command", "read_file", "edit_file", "write_file", "explore",
})

# The mini-swe-agent / quant-tuner shape: a shell and nothing else. Reading,
# editing, searching and testing all go through the same command interface,
# which is the tool vocabulary these models have seen most. Costs ~170 schema
# tokens against minimal's ~640 and standard's ~1850.
SHELL_ONLY_TOOLS = frozenset({"run_command"})

TOOL_PROFILES: dict[str, frozenset[str] | None] = {
    # None = no narrowing; the permission mode alone decides.
    "standard": None,
    "minimal": MINIMAL_TOOLS,
    "graph": GRAPH_TOOLS,
    "shell": SHELL_ONLY_TOOLS,
}

# Profiles with no dedicated edit tool: file changes necessarily go through
# the shell, so anything that gates the shell on "have you edited yet" would
# leave the model with no way to act at all.
PROFILES_WITHOUT_EDIT_TOOL = frozenset({"shell"})


def profile_has_edit_tool(profile: str) -> bool:
    return profile not in PROFILES_WITHOUT_EDIT_TOOL


def get_profile_tools(profile: str) -> frozenset[str] | None:
    """Return the tool-name filter for *profile*, or None for no narrowing."""
    return TOOL_PROFILES.get(profile)


def profile_shows(profile: str, tool_name: str) -> bool:
    """Would *tool_name* appear in *profile*'s schema?

    Exists so prose never names a tool the model cannot see. Every message
    the harness writes -- system prompt lines, tool-result hints, error
    recovery advice -- has to answer this before naming a tool, or it becomes
    another instruct-then-block: an instruction the model is unable to follow
    and has no way to discover why.
    """
    narrow = TOOL_PROFILES.get(profile)
    return narrow is None or tool_name in narrow


def get_allowed_tools(mode: str) -> frozenset[str]:
    """Return set of tool names allowed in given mode."""
    if mode == "yolo":
        return ALL_TOOLS
    if mode == "bench":
        return BENCH_TOOLS
    if mode == "edits":
        return READ_ONLY_TOOLS | MUTATING_TOOLS | SHELL_TOOL_NAMES
    return frozenset()


def check_permission(
    tool_name: str,
    mode: str,
    args: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (allowed, reason).

    `reason == "prompt"` means the caller should ask the user before executing.
    """
    # MCP tools: apply mode-based filtering.
    if tool_name.startswith("mcp__"):
        if mode == "edits":
            return False, "prompt"
        return True, ""

    allowed = get_allowed_tools(mode)

    if tool_name not in allowed:
        return False, f"refused: tool {tool_name} not available in {mode} mode"

    # Edits-mode shell prompt (only reached when tool IS in allowed set; for
    # run_command we also want the approval prompt to still fire).
    if mode == "edits" and tool_name in SHELL_TOOL_NAMES:
        return False, "prompt"

    return True, ""


