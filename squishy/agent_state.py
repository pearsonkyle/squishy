"""Agent data structures: TaskResult, LoopState, and message/loop helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from squishy.client import ToolCall


@dataclass
class TaskResult:
    success: bool
    final_text: str = ""
    turns_used: int = 0
    tokens_used: int = 0
    files_created: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    commands_run: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    plan_state: dict[str, Any] | None = None
    empty_responses: int = 0
    quality_skips: int = 0
    prose_completions: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    env_fix_files: list[str] = field(default_factory=list)
    edit_failures: int = 0
    # Phase/budget diagnostics.
    final_phase: str = ""
    explore_turns: int = 0
    fix_verify_cycles: int = 0
    total_quality_violations: int = 0
    # Per-turn event log for post-hoc analysis.
    turn_log: list[dict[str, Any]] = field(default_factory=list)
    # Complete message log (pre-trim) for SFT training data export.
    full_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopState:
    """Mutable counters shared across the agent loop and its sub-methods."""
    start: float
    consecutive_errors: int = 0
    plan_nudges: int = 0
    turns_without_plan_task: int = 0
    total_prompt_tokens: int = 0
    completion_tokens: int = 0
    files_created: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    commands_run: int = 0
    quality_retries: int = 0
    total_quality_violations: int = 0
    turns_without_progress: int = 0
    test_passed_after_edit: bool = False
    empty_responses: int = 0
    quality_skips: int = 0
    total_tool_calls: dict[str, int] = field(default_factory=dict)
    prose_completions: int = 0
    prior_created_len: int = 0
    prior_edited_len: int = 0
    # Phase-budget tracking (bench/yolo).
    phase: str = "explore"          # "explore" | "fix" | "verify"
    explore_turns: int = 0
    fix_verify_cycles: int = 0
    post_edit_read_turns: int = 0
    finish_countdown: int = -1      # -1 = inactive; N = force finish in N turns
    # Goal-drift tracking (bench/yolo).
    env_fix_files: set[str] = field(default_factory=set)
    problem_files: set[str] = field(default_factory=set)
    env_error_count: int = 0
    # Failed edit tracking.
    edit_failures_per_file: dict[str, int] = field(default_factory=dict)
    total_edit_failures: int = 0
    recent_edit_fail_files: set[str] = field(default_factory=set)
    # Re-anchoring.
    last_reanchor_turn: int = 0
    problem_text: str | None = None
    # FAIL_TO_PASS test identifiers for verifying the right tests are run.
    fail_to_pass_tests: list[str] = field(default_factory=list)
    # Compaction-resilient loop detection.
    last_call_key: str = ""
    consecutive_identical: int = 0
    unresolved_nudges: int = 0
    last_nudge_turn: int = -3
    total_nudges: int = 0
    # Test failure tracking across fix-verify cycles.
    last_test_failure_count: int = -1  # -1 = no test run yet
    last_test_failures: list[str] = field(default_factory=list)
    # Per-turn structured event log.
    turn_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def prose_msg(text: str, reasoning: str = "") -> dict[str, Any]:
    """Build a prose-only assistant message, preserving reasoning if present."""
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        msg["think"] = reasoning
    return msg


def assistant_msg(
    text: str, tool_calls: list[ToolCall], reasoning: str = "",
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)},
            }
            for tc in tool_calls
        ],
    }
    if reasoning:
        msg["think"] = reasoning
    return msg


def brief(tc: ToolCall) -> str:
    a = tc.args
    if tc.name in ("read_file", "write_file", "edit_file", "list_directory"):
        return str(a.get("path", ""))
    if tc.name == "search_files":
        return f'"{a.get("pattern", "")}"'
    if tc.name == "glob_files":
        return str(a.get("pattern", ""))
    if tc.name == "recall":
        return str(a.get("query", ""))
    return ""


def call_key(tool_calls: list[ToolCall]) -> str:
    """Build a stable key from a list of tool calls for loop detection."""
    parts = []
    for tc in tool_calls:
        try:
            args_str = json.dumps(tc.args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(tc.args)
        parts.append(f"{tc.name}:{args_str}")
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Problem-file extraction
# ---------------------------------------------------------------------------

_PY_PATH_RE = re.compile(r"(?:^|[\s\"'`(,])([a-zA-Z_][\w/]*\.py)\b")
_MODULE_RE = re.compile(r"(?:^|[\s\"'`(,])([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*){2,})\b")


def extract_problem_files(text: str) -> set[str]:
    """Extract likely file paths and module references from a problem statement."""
    paths: set[str] = set()
    for m in _PY_PATH_RE.finditer(text):
        paths.add(m.group(1).lower())
    for m in _MODULE_RE.finditer(text):
        parts = m.group(1).split(".")
        paths.add("/".join(parts).lower() + ".py")
        if len(parts) > 2:
            paths.add("/".join(parts[:-1]).lower() + ".py")
    return paths


def path_matches_problem(path: str, problem_files: set[str]) -> bool:
    """Check if an edited file path plausibly relates to the problem statement."""
    path_lower = path.lower().replace("\\", "/")
    for pf in problem_files:
        if pf in path_lower or path_lower.endswith(pf):
            return True
    base = os.path.basename(path_lower).replace(".py", "")
    return any(base in pf for pf in problem_files)


# ---------------------------------------------------------------------------
# Command classification
# ---------------------------------------------------------------------------

EXPLORE_TOOLS = frozenset({"read_file", "list_directory", "search_files", "glob_files", "recall"})
_TEST_CMD_KEYWORDS = ("pytest", "unittest", "python -m test", "python -m pytest", "test_")

EXPLORE_CMDS = frozenset({
    "grep", "rg", "sed", "cat", "head", "tail", "find", "awk", "wc", "od",
    "ls", "tree", "file", "stat", "less", "more",
})


def is_test_command(cmd: str) -> bool:
    """Return True if ``cmd`` looks like a test invocation."""
    return any(kw in cmd for kw in _TEST_CMD_KEYWORDS)


def test_covers_fail_to_pass(cmd: str, fail_to_pass: list[str]) -> bool:
    """Return True if *cmd* appears to run at least one FAIL_TO_PASS test.

    When ``fail_to_pass`` is empty, falls back to True (any test counts).
    This prevents false "test passed" declarations when the agent runs an
    unrelated test file that happens to pass.
    """
    if not fail_to_pass:
        return True
    # Normalise the command to check for test path/id overlap.
    cmd_lower = cmd.lower().replace("\\", "/")
    for test_id in fail_to_pass:
        # test_id is typically like "tests/test_foo.py::TestBar::test_baz"
        # or "test/test_foo.py::test_baz".
        # Check if any significant fragment appears in the command.
        tid = test_id.replace("\\", "/")
        # Try the full test id first.
        if tid in cmd:
            return True
        # Try just the test file path (before ::).
        parts = tid.split("::")
        test_file = parts[0]
        if test_file and test_file in cmd:
            return True
        # Try just the module path (e.g., "tests/test_foo" from "tests/test_foo.py")
        module = test_file.rsplit(".", 1)[0] if "." in test_file else test_file
        if module and module in cmd:
            return True
    return False


def is_exploration_command(cmd: str) -> bool:
    """Return True if ``cmd`` is a read-only exploration command."""
    text = cmd.strip()
    if not text:
        return False
    while text.startswith("cd "):
        for sep in ("&&", ";"):
            idx = text.find(sep)
            if idx != -1:
                text = text[idx + len(sep):].strip()
                break
        else:
            return False
    first = text.split()[0] if text else ""
    if first in EXPLORE_CMDS:
        return True
    # Catch python -c "open(...)" / "with open(...)" file-reading patterns
    if first in ("python3", "python") and "-c" in text:
        lower = text.lower()
        if any(kw in lower for kw in ("open(", "read(", "readlines(", "print(open")):
            return True
    return False
