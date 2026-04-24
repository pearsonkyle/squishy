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

    # 3. Repeated identical tool call (same name + same args as previous turn)
    if tool_calls:
        prev_calls = _extract_prev_tool_calls(messages)
        for tc in tool_calls:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
            args_json = _stable_json(args)
            for pname, pargs_json in prev_calls:
                if name == pname and args_json == pargs_json:
                    return False, "repeated_tool_call"

    # 4. Excessive same-file re-reads (same path+offset+limit 3+ times in last 4 turns)
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        if name == "read_file":
            args = getattr(tc, "args", {})
            read_key = (
                str(args.get("path", "")),
                args.get("offset"),
                args.get("limit"),
            )
            count = _count_recent_reads(messages, read_key, lookback=8)
            if count >= 2:  # already read twice, this would be the 3rd
                return False, "excessive_reread"

    # 5. Repeated run_command (same command in last 3 assistant turns)
    for tc in tool_calls:
        name = getattr(tc, "name", "")
        if name == "run_command":
            args = getattr(tc, "args", {})
            cmd = str(args.get("command", ""))
            if cmd and _count_recent_commands(messages, cmd, lookback=4) >= 2:
                return False, "repeated_command"

    # 6. Edit-verify loop: many consecutive edit->run_command cycles
    if tool_calls:
        cycle_count = _count_edit_verify_cycles(messages, lookback=12)
        if cycle_count >= 5:
            return False, "edit_verify_loop"

    return True, "ok"


def build_correction(reason: str) -> str:
    """Build a corrective system message based on the failure reason."""
    if reason.startswith("unknown_tool:"):
        tool_name = reason.split(":", 1)[1]
        return (
            f"Tool '{tool_name}' does not exist. Available tools are: "
            "read_file, write_file, edit_file, list_directory, search_files, "
            "glob_files, run_command, show_diff, save_note, recall. "
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
            "You just made the exact same tool call as your previous turn. "
            "This suggests you are stuck in a loop. If you have already made "
            "your fix and verified it works, STOP calling tools and respond with "
            "a plain text summary of what you changed and why. If you haven't "
            "fixed the bug yet, try a different approach: read a different file, "
            "adjust your edit_file old_str to match the actual file content, or "
            "re-read the problem statement."
        ),
        "excessive_reread": (
            "You have read this exact file range multiple times already. The content "
            "has not changed. Use `save_note` to record important content so you "
            "remember it across long conversations. If you are looking for a different "
            "file, use `recall(query=...)` to search the index instead of re-reading. "
            "Proceed with your fix using the content you already have."
        ),
        "repeated_command": (
            "You have already run this exact command recently. Running the same "
            "verification command multiple times is wasteful. If the test passed, "
            "you are DONE — respond with a plain text summary of what you changed "
            "and why. Do NOT call any more tools. If the test failed, fix the code "
            "with `edit_file` instead of re-running the same command."
        ),
        "edit_verify_loop": (
            "You have been cycling between edit_file and run_command for many turns "
            "without resolving the issue. STOP and try a completely different approach: "
            "1. Re-read the problem statement carefully. "
            "2. Consider if you are editing the wrong file or function. "
            "3. If the same test keeps failing, re-read the test to understand expectations. "
            "If your fix is working (tests pass), respond with a plain text summary immediately."
        ),
    }
    return corrections.get(reason, f"Quality issue detected: {reason}. Please try again.")



def _extract_prev_tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract (name, normalized_args_json) pairs from the SECOND most recent
    assistant tool calls (skipping the current turn which was just appended).

    Args are re-serialized via ``_stable_json`` so key ordering matches the
    caller's comparison in ``assess_response``.
    """
    found_first = False
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if not found_first:
                found_first = True
                continue  # skip the current turn
            out = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                raw = func.get("arguments", "{}")
                # Normalize: parse then re-serialize with sorted keys.
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    parsed = raw
                out.append((name, _stable_json(parsed)))
            return out
    return []


def _stable_json(d: dict[str, Any] | Any) -> str:
    """Serialize a dict to JSON with sorted keys for stable comparison."""
    if not isinstance(d, dict):
        return str(d)
    try:
        return json.dumps(d, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(d)


def _count_recent_reads(
    messages: list[dict[str, Any]],
    read_key: tuple[str, Any, Any],
    lookback: int = 8,
) -> int:
    """Count how many times the same read_file(path, offset, limit) appears
    in the last `lookback` assistant messages."""
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
    # Collect per-turn tool name sets (most recent first).
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

    # Count consecutive turns where either:
    #   - The turn itself has both edit_file and run_command, or
    #   - Adjacent turns alternate between edit_file-only and run_command-only.
    cycles = 0
    i = 0
    while i < len(turn_tools):
        tools = turn_tools[i]
        if "edit_file" in tools and "run_command" in tools:
            # Single turn with both = 1 cycle
            cycles += 1
            i += 1
        elif "edit_file" in tools and i + 1 < len(turn_tools) and "run_command" in turn_tools[i + 1]:
            cycles += 1
            i += 2
        elif "run_command" in tools and i + 1 < len(turn_tools) and "edit_file" in turn_tools[i + 1]:
            cycles += 1
            i += 2
        else:
            break  # non-cycle turn breaks the streak
    return cycles
