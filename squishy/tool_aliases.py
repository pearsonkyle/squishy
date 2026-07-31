"""Tolerance layer for tool-call vocabulary.

Small local models are often fine-tuned against a *different* tool vocabulary
than squishy's (Anthropic's ``bash`` / ``str_replace_based_edit_tool`` /
``view`` / ``create``, OpenAI/Aider/SWE-agent names, etc.). When such a model
emits ``bash`` instead of ``run_command`` or ``file_path`` instead of ``path``,
squishy would otherwise reject the call as an unknown tool / missing param and
burn a turn. This module maps well-established alternate tool NAMES and
parameter names onto squishy's canonical tools, so that fine-tuned knowledge
transfers instead of fighting the harness.

It is a *forgiveness* layer, applied once to each tool call before loop
checks and dispatch:
  * a canonical name is never remapped;
  * an alias parameter is only applied when the canonical parameter is absent
    (an explicit canonical value always wins);
  * only unambiguous, widely-used aliases are included — nothing that could
    plausibly mean two different squishy tools/params.
"""

from __future__ import annotations

from typing import Any

# Alternate tool names → squishy canonical tool name. Kept conservative:
# every entry is a name a mainstream coding-agent harness actually uses, and
# maps unambiguously to exactly one squishy tool.
TOOL_NAME_ALIASES: dict[str, str] = {
    # shell
    "bash": "run_command",
    "shell": "run_command",
    "sh": "run_command",
    "execute_bash": "run_command",
    "execute_command": "run_command",
    "run_shell": "run_command",
    "run_bash": "run_command",
    "terminal": "run_command",
    # read
    # `read`/`edit`/`write` are Claude Code's tool names (matched
    # case-insensitively, so `Read` lands here too). They were missing while
    # its `Bash`/`Glob`/`Grep`/`LS` were covered, so a Claude-Code-trained
    # model got its shell call forgiven and its file calls rejected.
    "read": "read_file",
    "view": "read_file",
    "cat": "read_file",
    "open_file": "read_file",
    "view_file": "read_file",
    # write / create
    "write": "write_file",
    "create": "write_file",
    "create_file": "write_file",
    "new_file": "write_file",
    # edit (Anthropic text-editor + common synonyms)
    "edit": "edit_file",
    "str_replace": "edit_file",
    "str_replace_editor": "edit_file",
    "str_replace_based_edit_tool": "edit_file",
    "replace_in_file": "edit_file",
    "apply_edit": "edit_file",
    # list
    "ls": "list_directory",
    "list_dir": "list_directory",
    "list_files": "list_directory",
    # search
    "grep": "search_files",
    "ripgrep": "search_files",
    "search_code": "search_files",
    "grep_search": "search_files",
    # glob
    "glob": "glob_files",
    "find_files": "glob_files",
    "file_search": "glob_files",
    # diff / notes / web
    "git_diff": "show_diff",
    "diff": "show_diff",
    "remember": "save_note",
    "fetch": "fetch_url",
    "web_fetch": "fetch_url",
}

# Per-canonical-tool parameter aliases → canonical parameter name.
# Scoped per tool so the same alias can mean different things for different
# tools (``query`` is canonical for recall but an alias for search's pattern).
_PATH = {"file_path": "path", "filepath": "path", "filename": "path", "file": "path"}
_DIR = {"dir": "path", "directory": "path", "folder": "path", "file_path": "path", "filepath": "path"}

PARAM_ALIASES: dict[str, dict[str, str]] = {
    "read_file": _PATH,
    "write_file": {
        **_PATH,
        "text": "content", "file_text": "content", "data": "content",
        "body": "content", "contents": "content",
    },
    "edit_file": {
        **_PATH,
        "old_string": "old_str", "old_text": "old_str", "original": "old_str", "search": "old_str",
        "new_string": "new_str", "new_text": "new_str", "replacement": "new_str", "replace": "new_str",
    },
    "list_directory": _DIR,
    "search_files": {
        "regex": "pattern", "query": "pattern", "search": "pattern",
        "search_pattern": "pattern", "text": "pattern",
        "dir": "path", "directory": "path",
    },
    "glob_files": {
        "glob": "pattern", "pattern_glob": "pattern",
        "dir": "path", "directory": "path",
    },
    "run_command": {
        "cmd": "command", "shell_command": "command", "bash_command": "command",
        "script": "command", "commands": "command",
    },
    "recall": {"q": "query", "question": "query", "search": "query", "text": "query"},
    "fetch_url": {"uri": "url", "link": "url", "address": "url"},
    "save_note": {
        "name": "key", "title": "key", "label": "key",
        "note": "content", "value": "content", "text": "content",
    },
}


def canonical_tool_name(name: str) -> str:
    """Map an alias tool name to its canonical squishy name (identity if none)."""
    if not isinstance(name, str):
        return name
    if name in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[name]
    # Case-insensitive fallback for models that emit e.g. "Bash" / "Grep".
    lowered = name.lower()
    return TOOL_NAME_ALIASES.get(lowered, name)


def normalize_args(canonical_name: str, args: Any) -> Any:
    """Remap alias parameter names to canonical ones for *canonical_name*.

    Only remaps when the canonical key is absent, so an explicit canonical
    value is never clobbered by an alias. Non-dict args pass through untouched
    (the client's ``_tool_arg_error`` sentinel must survive).
    """
    amap = PARAM_ALIASES.get(canonical_name)
    if not amap or not isinstance(args, dict):
        return args
    if not any(k in amap for k in args):
        return args  # nothing to remap — avoid rebuilding the dict
    out: dict[str, Any] = {}
    for k, v in args.items():
        target = amap.get(k, k)
        # Preserve a preserved sentinel / an already-present canonical key.
        if target != k and (target in args or target in out):
            out[k] = v  # canonical already supplied — keep alias under its own name
        else:
            out[target] = v
    return out


def normalize_call(name: str, args: Any) -> tuple[str, Any]:
    """Return the (canonical_name, canonical_args) for a raw tool call."""
    canon = canonical_tool_name(name)
    return canon, normalize_args(canon, args)


__all__ = [
    "TOOL_NAME_ALIASES",
    "PARAM_ALIASES",
    "canonical_tool_name",
    "normalize_args",
    "normalize_call",
]
