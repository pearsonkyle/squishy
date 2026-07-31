"""Agent data structures: TaskResult, LoopState, and message/loop helpers."""

from __future__ import annotations

import json
import os
import re
import shlex
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
    empty_responses: int = 0
    prose_completions: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    edit_failures: int = 0
    # Per-turn event log for post-hoc analysis.
    turn_log: list[dict[str, Any]] = field(default_factory=list)
    # Complete message log (pre-trim) for SFT training data export.
    full_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopState:
    """Mutable counters shared across the agent loop and its sub-methods."""
    start: float
    consecutive_errors: int = 0
    total_prompt_tokens: int = 0
    completion_tokens: int = 0
    files_created: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    commands_run: int = 0
    empty_responses: int = 0
    total_tool_calls: dict[str, int] = field(default_factory=dict)
    prose_completions: int = 0
    problem_files: set[str] = field(default_factory=set)
    total_edit_failures: int = 0
    # Shell commands that looked like they modified a file. Under a shell-only
    # profile this is the only edit signal there is — `files_edited` only ever
    # records edit_file/write_file.
    shell_writes: int = 0
    recent_edit_fail_files: set[str] = field(default_factory=set)
    last_edit_turn: int = 0
    # FAIL_TO_PASS identifiers, when a bench harness supplies them. Surfaced to
    # the model through the prompt; the loop itself no longer gates on them.
    fail_to_pass_tests: list[str] = field(default_factory=list)
    # Harness-supplied test command (SWE-rebench V2's install_config.test_cmd).
    test_cmd: str = ""
    # Compaction-resilient loop detection.
    last_call_key: str = ""
    consecutive_identical: int = 0
    # Nudge budget.
    last_nudge_turn: int = -3
    total_nudges: int = 0
    nudges_this_turn: int = 0
    # Per-turn structured event log.
    turn_log: list[dict[str, Any]] = field(default_factory=list)
    compaction_count: int = 0
    # LLM error count — used to retry transient failures in bench mode.
    llm_errors: int = 0
    # Cumulative tenacity retries across all completion calls in this loop.
    cumulative_retries: int = 0


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


def is_test_path(path: str) -> bool:
    """Return True when *path* looks like a test module/dir.

    Used by the dispatch layer to gate plan-approval EXECUTE nudges on
    whether the agent has actually read at least one failing-test file.
    """
    pl = path.lower()
    return (
        "/test_" in pl or pl.startswith("test_")
        or "/tests/" in pl or pl.startswith("tests/")
        or "/test/" in pl or pl.startswith("test/")
    )


# ---------------------------------------------------------------------------
# Command classification
# ---------------------------------------------------------------------------

# Runner flags that inspect/collect rather than actually run the suite — an
# exit-0 for one of these must NOT count as "tests passed".
_META_RUNNER_FLAGS = frozenset({
    "--version", "--help", "-h", "--collect-only", "--co", "--fixtures", "--markers",
})


def _runner_args(cmd: str) -> list[str] | None:
    """If any &&/;/| segment is a genuine test-runner invocation — the runner as
    the segment's leading token(s), not merely the word "pytest" appearing
    inside a string — return the runner's argument tokens (everything after the
    runner spec). Otherwise None.

    This is what stops ``git commit -m "fix pytest failure"`` or
    ``echo running pytest`` from being classified as a passing test run.
    """
    for seg in re.split(r"&&|\|\||;|\|", cmd):
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg)
        except ValueError:
            continue
        if not toks:
            continue
        low = [t.lower() for t in toks]
        if low[0] == "pytest":
            return toks[1:]
        if low[0] in ("python", "python3"):
            if len(toks) >= 3 and low[1] == "-m" and low[2] in ("pytest", "unittest"):
                return toks[3:]
            # `python test_x.py` / `python ./test_x.py` — the script is the target.
            if len(toks) >= 2 and os.path.basename(toks[1]).startswith("test_"):
                return toks[1:]
    return None


def is_test_command(cmd: str) -> bool:
    """Return True if ``cmd`` is a genuine test-runner invocation.

    Requires a real runner as a segment's leading token — merely having
    ``test_`` in a path (``ls tests/``) or ``pytest`` inside a commit message
    does not count.
    """
    return _runner_args(cmd) is not None


def test_covers_fail_to_pass(cmd: str, fail_to_pass: list[str]) -> bool:
    """Return True if *cmd* appears to run at least one FAIL_TO_PASS test.

    When ``fail_to_pass`` is empty, falls back to True (any test counts).
    This prevents false "test passed" declarations when the agent runs an
    unrelated test file that happens to pass.
    """
    if not fail_to_pass:
        return True
    return bool(f2p_files_in_command(cmd, fail_to_pass))


def distinct_f2p_files(fail_to_pass: list[str]) -> set[str]:
    """Return the set of unique test file paths in ``fail_to_pass``.

    F2P ids are like ``tests/test_xray_2d.py::test_matched_adjoint`` —
    we strip the ``::`` suffix and any parametrize ``[...]`` tail, then
    return the file path normalised to forward slashes.
    """
    out: set[str] = set()
    for tid in fail_to_pass:
        if "::" in tid:
            path = tid.split("::", 1)[0].replace("\\", "/").strip()
        else:
            # Non-pytest / bare id (e.g. terminal-bench "test_foo"): use the id
            # itself as the selector so coverage can still be satisfied — the
            # old code skipped these, leaving the gate permanently unmet.
            path = tid.replace("\\", "/").strip()
        if path:
            out.add(path)
    return out


def f2p_files_in_command(cmd: str, fail_to_pass: list[str]) -> set[str]:
    """Return the F2P test files that this command appears to exercise.

    Matches the same heuristics as ``test_covers_fail_to_pass`` but
    returns the set of *covered files* rather than a bool — F5 needs
    file-level coverage tracking to gate ``finish_plan`` correctly when
    F2P spans multiple test files (e.g. scico-561's 2D + 3D tests).

    A bare ``pytest`` with no path argument is conservatively treated
    as covering every F2P file (it exercises the whole suite).
    """
    files = distinct_f2p_files(fail_to_pass)
    if not files:
        return set()

    args = _runner_args(cmd)
    if args is None:
        # Not a genuine test-runner invocation (e.g. `git commit -m "…pytest…"`).
        return set()
    # A meta invocation (`pytest --version` / `--collect-only`) did not run the
    # suite, so it covers nothing — this closes the exit-0 false-positive.
    if any(a.lower() in _META_RUNNER_FLAGS for a in args):
        return set()
    # Bare runner with no positional path arg — assume it ran the whole suite.
    if not any(not a.startswith("-") for a in args):
        return set(files)

    cmd_norm = cmd.replace("\\", "/")
    covered: set[str] = set()
    for path in files:
        if path in cmd_norm:
            covered.add(path)
            continue
        # Module form: tests/test_foo (no .py) — also acceptable.
        module = path.rsplit(".", 1)[0] if "." in path else path
        if module and module in cmd_norm:
            covered.add(path)
    return covered




# Shell fragments that indicate a command wrote to a file. Deliberately broad:
# a false positive only means we skip one nudge, while a false negative means
# nagging a model that has already done the work.
_WRITE_CMD_RE = re.compile(
    r"(?x)"
    r"(?<![0-9])>>?\s*[\w./~$-]"       # `> file` / `>> file`, not `2>&1`
    r"|\b(?:sed|perl)\b[^|;&]*\s-i"    # in-place edit
    r"|\btee\b"
    r"|\bpatch\b\s+-p\d"
    r"|\bgit\s+apply\b"
    r"|\bmv\b|\bcp\b"
    r"|\.write\(|\.write_text\(|open\([^)]*['\"][wa]"
)


def looks_like_file_write(command: str) -> bool:
    """True if *command* plausibly modified a file on disk.

    Used only to decide whether the agent still needs prodding toward making
    an edit; it never gates or blocks anything, so over-matching is cheap.
    """
    if not isinstance(command, str) or not command.strip():
        return False
    return bool(_WRITE_CMD_RE.search(command))
