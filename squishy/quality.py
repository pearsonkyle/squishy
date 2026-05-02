"""Response quality monitoring for the agent loop.

Heuristic checks to catch common failure modes before they waste turns:
- Repeated identical tool calls (infinite loop)
- Hallucinated / unknown tool names
- Malformed tool call arguments
- Excessive re-reads of the same file

Quality checks are cheap (no LLM calls) and run after every assistant turn.
Ported from little-coder/local/quality.py with squishy-native types.
"""
from __future__ import annotations

import json
import shlex
from typing import Any

from squishy.tools import REGISTRY


def assess_response(
    tool_calls: list[Any],
    messages: list[dict[str, Any]],
    registry: dict[str, Any],
) -> tuple[bool, str]:
    """Heuristic quality check on an assistant response's tool calls.

    Args:
        tool_calls: list of ToolCall objects (have .name and .args attrs).
        messages: full conversation history.
        registry: dict mapping tool name -> Tool object.

    Returns:
        (ok, reason) — True if response is acceptable, False + reason if not.
    """
    for tc in tool_calls:
        name = getattr(tc, "name", "") or ""
        args = getattr(tc, "args", {}) or {}

        # 1. Hallucinated tool name
        if name and name not in registry:
            return False, f"unknown_tool:{name}"

        # 2. Malformed args (JSON parse failed in client.py)
        if isinstance(args, dict) and args.get("_tool_arg_error"):
            return False, f"malformed_args:{name}"

    # 2b. Duplicate tool calls within the same turn
    if len(tool_calls) >= 2:
        seen_keys: set[str] = set()
        for tc in tool_calls:
            name = getattr(tc, "name", "") or ""
            args = getattr(tc, "args", {}) or {}
            key = f"{name}:{_stable_json(args)}"
            if key in seen_keys:
                return False, "repeated_tool_call"
            seen_keys.add(key)

    # 3. Repeated identical tool call (same name + same args within last 3 turns)
    #    Exception: run_command is allowed to repeat if an edit_file/write_file
    #    occurred between the current turn and the matching turn (post-edit test).
    if tool_calls:
        recent_turns = _extract_recent_tool_calls(messages, lookback=3)
        for tc in tool_calls:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
            args_json = _stable_json(args)
            for turn_idx, turn_calls in enumerate(recent_turns):
                for pname, pargs_json in turn_calls:
                    if name == pname and args_json == pargs_json:
                        # Allow run_command to repeat after an edit
                        if name == "run_command" and _edit_between_turns(
                            messages, turns_back=turn_idx + 1,
                        ):
                            continue
                        return False, "repeated_tool_call"

    # 4. Excessive same-file re-reads
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        if name == "read_file":
            args = getattr(tc, "args", {})
            # 4a. Exact same (path, offset, limit) read 2+ times in 8 turns
            read_key = (
                str(args.get("path", "")),
                args.get("offset"),
                args.get("limit"),
            )
            count = _count_recent_reads(messages, read_key, lookback=8)
            if count >= 2:
                return False, "excessive_reread"
            # 4b. Same file path (any range) read 3+ times in 10 turns
            path_key = (str(args.get("path", "")), None, None)
            path_count = _count_recent_reads(
                messages, path_key, lookback=10, match_path_only=True,
            )
            if path_count >= 4:
                return False, "excessive_reread"

    # 5. Repeated run_command (same command in last 3 assistant turns)
    #    Exception: allowed to repeat if an edit occurred between the two runs.
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        if name == "run_command":
            args = getattr(tc, "args", {})
            cmd = str(args.get("command", ""))
            if cmd and _count_recent_commands(messages, cmd, lookback=4) >= 2:
                if not _edit_between_commands(messages, cmd, lookback=4):
                    return False, "repeated_command"

    # 6. Edit-verify loop: many consecutive edit->run_command cycles
    if tool_calls:
        cycle_count = _count_edit_verify_cycles(messages, lookback=12)
        if cycle_count >= 5:
            return False, "edit_verify_loop"

    # 7. Repeated recall (same query in last 4 assistant turns)
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        if name == "recall":
            args = getattr(tc, "args", {})
            query = str(args.get("query", ""))
            if query and _count_recent_tool_with_arg(messages, "recall", "query", query, lookback=4) >= 2:
                return False, "repeated_recall"

    # 8. Repeated search_files (same pattern in last 4 assistant turns)
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        if name == "search_files":
            args = getattr(tc, "args", {})
            pattern = str(args.get("pattern", ""))
            if pattern and _count_recent_tool_with_arg(
                messages, "search_files", "pattern", pattern, lookback=4
            ) >= 2:
                return False, "repeated_search"

    # 9. Excessive search_files usage (any pattern, 6+ in last 10 turns)
    for tc in tool_calls:
        if getattr(tc, "name", "") == "search_files":
            total_searches = _count_recent_tool_calls_by_name(messages, "search_files", lookback=10)
            if total_searches >= 6:
                return False, "repeated_search"
            break

    return True, "ok"


def build_correction(reason: str) -> str:
    """Build a corrective system message based on the failure reason."""
    if reason.startswith("unknown_tool:"):
        tool_name = reason.split(":", 1)[1]
        available = ", ".join(REGISTRY.keys())
        return (
            f"Tool '{tool_name}' does not exist. Available tools are: "
            f"{available}. "
            "Use one of these instead."
        )

    if reason.startswith("malformed_args:"):
        tool_name = reason.split(":", 1)[1]
        return (
            f"The arguments for tool '{tool_name}' were malformed (not valid JSON). "
            "Please provide the arguments as a proper JSON object with correct syntax."
        )

    corrections = {
        "repeated_tool_call": (
            "You just made the exact same tool call as a recent turn. "
            "This suggests you are stuck in a loop. If you have already completed "
            "your task, STOP calling tools and respond with a plain text summary. "
            "If not, try a different approach: use a different tool, adjust your "
            "arguments, or reconsider your strategy."
        ),
        "excessive_reread": (
            "BLOCKED: You have read this exact file multiple times. The content has "
            "NOT changed. You MUST stop reading and take action NOW: call `edit_file` "
            "to make your fix, or call `run_command` to test. Do NOT call `read_file` "
            "on this file again."
        ),
        "repeated_command": (
            "You have already run this exact command recently without changing any "
            "code in between. If it failed, edit the code first, THEN re-run. "
            "If it succeeded, you are DONE — respond with a plain text summary."
        ),
        "edit_verify_loop": (
            "You have been cycling between edit_file and run_command for many turns "
            "without resolving the issue. STOP and try a completely different approach: "
            "1. Re-read the requirements carefully. "
            "2. Consider if you are editing the wrong file or section. "
            "3. If the same check keeps failing, re-read it to understand expectations. "
            "If your changes are working, respond with a plain text summary immediately."
        ),
        "repeated_recall": (
            "You have searched for the same query multiple times. The results will not "
            "change. Use the results you already have, try a different query, or use "
            "`save_note` to record important findings. Try a different approach."
        ),
        "repeated_search": (
            "You have searched for the same pattern multiple times. The results will not "
            "change. Use the results you already have or try a different search pattern."
        ),
    }
    return corrections.get(reason, f"Quality issue detected: {reason}. Please try again.")



def _extract_recent_tool_calls(
    messages: list[dict[str, Any]], lookback: int = 3,
) -> list[list[tuple[str, str]]]:
    """Extract tool calls from the last ``lookback`` assistant turns.

    Skips the current turn (most recent assistant message). Returns a list of
    turns, where each turn is a list of (name, normalized_args_json) pairs.
    """
    turns: list[list[tuple[str, str]]] = []
    found_current = False
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if not found_current:
                found_current = True
                continue  # skip the current turn
            calls: list[tuple[str, str]] = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                raw = func.get("arguments", "{}")
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = raw
                calls.append((name, _stable_json(parsed)))
            turns.append(calls)
            if len(turns) >= lookback:
                break
    return turns


def _stable_json(d: dict[str, Any] | Any) -> str:
    """Serialize a dict to JSON with sorted keys for stable comparison."""
    if not isinstance(d, dict):
        return str(d)
    try:
        return json.dumps(d, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(d)


_EDIT_TOOLS = frozenset({"edit_file", "write_file"})


def _edit_between_turns(
    messages: list[dict[str, Any]],
    turns_back: int,
) -> bool:
    """Check if an edit_file or write_file call occurred between the current
    assistant turn and ``turns_back`` assistant turns ago.

    ``turns_back=1`` means between the previous and current assistant turns.
    """
    seen_assistant = 0
    found_current = False
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if not found_current:
                found_current = True
                continue
            seen_assistant += 1
            if seen_assistant > turns_back:
                break
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                if func.get("name", "") in _EDIT_TOOLS:
                    return True
    return False


def _edit_between_commands(
    messages: list[dict[str, Any]],
    command: str,
    lookback: int = 4,
) -> bool:
    """Check if an edit_file/write_file occurred between two occurrences of
    the same ``run_command`` in the last ``lookback`` assistant messages.

    Returns True if an edit was found between the current (most recent)
    run_command and any earlier matching run_command, meaning the repeat
    is a legitimate post-edit verify.
    """
    norm = _normalize_command(command)
    seen_assistant = 0
    found_current = False
    saw_edit = False
    for msg in reversed(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        seen_assistant += 1
        if seen_assistant > lookback:
            break
        has_matching_cmd = False
        has_edit = False
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            if name in _EDIT_TOOLS:
                has_edit = True
            if name == "run_command":
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                if _normalize_command(args.get("command", "")) == norm:
                    has_matching_cmd = True
        if has_matching_cmd and not found_current:
            found_current = True
            continue  # skip the current turn
        if has_edit:
            saw_edit = True
        if has_matching_cmd and found_current:
            # Found a previous matching command — was there an edit between?
            return saw_edit
    return False


def _count_recent_reads(
    messages: list[dict[str, Any]],
    read_key: tuple[str, Any, Any],
    lookback: int = 8,
    match_path_only: bool = False,
) -> int:
    """Count how many times read_file appears in the last `lookback` assistant messages.

    If ``match_path_only`` is True, only the path component is compared
    (ignoring offset/limit), which catches repeated reads of the same file
    with slightly different ranges.
    """
    count = 0
    seen_assistant = 0
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            seen_assistant += 1
            if seen_assistant > lookback:
                break
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if func.get("name") != "read_file":
                    continue
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                if match_path_only:
                    if str(args.get("path", "")) == read_key[0]:
                        count += 1
                else:
                    key = (
                        str(args.get("path", "")),
                        args.get("offset"),
                        args.get("limit"),
                    )
                    if key == read_key:
                        count += 1
    return count


def _normalize_command(cmd: str) -> str:
    """Normalize a shell command for fuzzy comparison.

    Expands combined short flags (``-xvs`` → ``-x -v -s``), sorts flags,
    and strips whitespace so that e.g. ``pytest tests/foo.py -xvs`` and
    ``pytest tests/foo.py -x -v -s`` are treated as the same command.
    """
    try:
        tokens = shlex.split(cmd.strip())
    except ValueError:
        return cmd.strip()
    if not tokens:
        return cmd.strip()
    binary: list[str] = [tokens[0]]
    flags: set[str] = set()
    positionals: list[str] = []
    for tok in tokens[1:]:
        if tok.startswith("--"):
            flags.add(tok)
        elif tok.startswith("-") and len(tok) > 1:
            # Expand combined short flags: -xvs -> {-x, -v, -s}
            for ch in tok[1:]:
                flags.add(f"-{ch}")
        else:
            positionals.append(tok)
    return " ".join(binary + sorted(flags) + positionals)


def _count_recent_commands(
    messages: list[dict[str, Any]],
    command: str,
    lookback: int = 4,
) -> int:
    """Count how many times the same run_command(command=...) appears
    in the last `lookback` assistant messages.  Uses normalized comparison."""
    norm = _normalize_command(command)
    count = 0
    seen_assistant = 0
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            seen_assistant += 1
            if seen_assistant > lookback:
                break
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if func.get("name") != "run_command":
                    continue
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                if _normalize_command(args.get("command", "")) == norm:
                    count += 1
    return count


def _count_edit_verify_cycles(
    messages: list[dict[str, Any]],
    lookback: int = 12,
) -> int:
    """Count consecutive (edit_file, run_command) pair cycles in recent history.

    Walks backward through the last ``lookback`` assistant messages and counts
    how many consecutive turns alternate between edit_file and run_command
    (in any order within the same turn — we check if the turn contains at
    least one edit_file AND at least one run_command, or the pattern across
    adjacent turns).
    """
    # Collect per-turn tool name sets in chronological order (oldest first).
    turn_tools: list[set[str]] = []
    seen = 0
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            names = set()
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                if name:
                    names.add(name)
            turn_tools.append(names)
            seen += 1
            if seen >= lookback:
                break
    turn_tools.reverse()  # oldest first so "look ahead" finds later turns

    # Count consecutive turns where either:
    #   - The turn itself has both edit_file and run_command, or
    #   - Adjacent turns alternate between edit_file-only and run_command-only.
    # Read-only turns (read_file, list_directory, search_files, etc.) are
    # allowed between edit/command turns without breaking the streak.
    _READ_ONLY = {"read_file", "list_directory", "search_files", "glob_files", "recall"}
    cycles = 0
    i = 0
    while i < len(turn_tools):
        tools = turn_tools[i]
        if "edit_file" in tools and "run_command" in tools:
            cycles += 1
            i += 1
        elif "edit_file" in tools:
            # Look ahead past read-only turns for a matching run_command.
            j = i + 1
            while j < len(turn_tools) and turn_tools[j].issubset(_READ_ONLY):
                j += 1
            if j < len(turn_tools) and "run_command" in turn_tools[j]:
                cycles += 1
                i = j + 1
            else:
                break
        elif "run_command" in tools:
            j = i + 1
            while j < len(turn_tools) and turn_tools[j].issubset(_READ_ONLY):
                j += 1
            if j < len(turn_tools) and "edit_file" in turn_tools[j]:
                cycles += 1
                i = j + 1
            else:
                break
        elif tools.issubset(_READ_ONLY):
            # Pure read-only turn — skip without breaking the streak.
            i += 1
        else:
            break
    return cycles


def _count_recent_tool_calls_by_name(
    messages: list[dict[str, Any]],
    tool_name: str,
    lookback: int = 10,
) -> int:
    """Count how many times ``tool_name`` was called in the last ``lookback`` assistant messages."""
    count = 0
    seen_assistant = 0
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            seen_assistant += 1
            if seen_assistant > lookback:
                break
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if func.get("name") == tool_name:
                    count += 1
    return count


def _count_recent_tool_with_arg(
    messages: list[dict[str, Any]],
    tool_name: str,
    arg_key: str,
    arg_value: str,
    lookback: int = 4,
) -> int:
    """Count how many times ``tool_name`` was called with ``arg_key==arg_value``
    in the last ``lookback`` assistant messages."""
    count = 0
    seen_assistant = 0
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            seen_assistant += 1
            if seen_assistant > lookback:
                break
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if func.get("name") != tool_name:
                    continue
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue
                if str(args.get(arg_key, "")) == arg_value:
                    count += 1
    return count
