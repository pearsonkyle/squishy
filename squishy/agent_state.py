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
    test_passed_after_edit: bool = False
    empty_responses: int = 0
    quality_skips: int = 0
    total_tool_calls: dict[str, int] = field(default_factory=dict)
    prose_completions: int = 0
    # Phase tracking (bench/yolo) — synced from PhaseState in bench mode.
    phase: str = "explore"
    explore_turns: int = 0
    fix_verify_cycles: int = 0
    # Goal-drift tracking (bench/yolo).
    env_fix_files: set[str] = field(default_factory=set)
    problem_files: set[str] = field(default_factory=set)
    env_error_count: int = 0
    # Failed edit tracking.
    edit_failures_per_file: dict[str, int] = field(default_factory=dict)
    total_edit_failures: int = 0
    recent_edit_fail_files: set[str] = field(default_factory=set)
    # Identical-old_str-per-path tracking (catches edit loops that bypass
    # repeated_tool_call detection because args.new_str varies). Maps
    # `path -> (last_old_str_hash, consecutive_count)`.
    last_edit_old_str_per_file: dict[str, tuple[str, int]] = field(default_factory=dict)
    # Re-anchoring.
    last_reanchor_turn: int = 0
    problem_text: str | None = None
    # FAIL_TO_PASS test identifiers for verifying the right tests are run.
    fail_to_pass_tests: list[str] = field(default_factory=list)
    # v6e: harness-supplied test command (V2's install_config.test_cmd).
    # Empty string when not provided (V1, terminal-bench, interactive REPL).
    # Consumed by maybe_post_edit_pytest_nudge to suggest the right runner
    # for non-pytest projects (npm/phpunit/cargo) instead of hardcoding pytest.
    test_cmd: str = ""
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
    # Compaction count — used to detect loops that survive compaction.
    compaction_count: int = 0
    # LLM error count — used to retry transient failures in bench mode.
    llm_errors: int = 0
    # Cumulative tenacity retries across all completion calls in this loop.
    # Read from `client.last_call_retries` after each completion.  When this
    # crosses a threshold the agent injects a CRITICAL "wrap up now" nudge
    # rather than letting upstream instability silently consume the budget.
    cumulative_retries: int = 0
    # F2P finish-plan gate intercepts.  In bench mode, finish_plan is
    # blocked once when the agent declares done without having actually
    # passed the FAIL_TO_PASS tests.  After one intercept the gate releases
    # so a degraded test environment cannot trap the agent forever.
    f2p_finish_gate_intercepts: int = 0
    # Last F2P-only failure count for cross-cycle progress tracking.
    last_f2p_failure_count: int = -1
    # Single-shot prose-completion gate.  When the agent prose-completes
    # in bench/yolo mode after observing a test failure but without making
    # any edits to fix it, the loop intercepts once and nudges the agent
    # to either fix or call finish_plan(status="failure").  Released after
    # one intercept so genuinely unfixable runs can still terminate.
    no_progress_intercepts: int = 0
    # F5: F2P file-coverage tracking.  ``f2p_files_covered`` records which
    # FAIL_TO_PASS test files have been exercised by a passing test command
    # (exit 0, no failures in test_summary).  ``test_passed_after_edit`` is
    # only allowed to flip True when *every* distinct F2P file has been
    # covered.  The gate intercepts ``finish_plan`` calls otherwise, with
    # the same single-shot release as A5 so degraded environments don't
    # trap the agent.  v27.2 fix for the scico-561 case where the agent
    # ran the 2D test, saw it pass, and skipped the 3D test entirely.
    f2p_files_covered: set[str] = field(default_factory=set)
    f2p_coverage_intercepts: int = 0
    # v2 auto-pytest finish gate.  ``last_edit_turn`` stamps the turn of the
    # most recent successful edit_file/write_file; ``last_f2p_test_turn``
    # stamps the turn of the most recent run_command pytest that hit any
    # F2P file (regardless of pass/fail).  When the agent tries to finish
    # in bench mode and ``last_edit_turn > last_f2p_test_turn``, the harness
    # synthesizes a pytest run on the F2P tests so the model gets one more
    # chance to react to real test output.  ``auto_pytest_runs`` caps the
    # number of times the gate fires per instance.
    last_edit_turn: int = 0
    last_f2p_test_turn: int = 0
    auto_pytest_runs: int = 0
    # v5 pre-finish gate: failing F2P test details from the most recent
    # F2P run.  Populated from ``test_summary`` when the agent's pytest
    # output parses cleanly.  Empty list when the last F2P run didn't
    # fail, the test runner isn't pytest, or no F2P run has happened yet.
    # Each entry: ``{"test": "<nodeid>", "error": "<truncated assertion>"}``.
    last_f2p_failures: list[dict[str, str]] = field(default_factory=list)
    # v6b pre-finish gate: True when the most recent F2P-covering test
    # command exited with collection / import errors but produced no
    # parseable per-test failure lines.  Lets the gate emit a "fix
    # collection before finishing" hint instead of falling silently
    # through to legacy branches when ``last_f2p_failures`` is empty.
    last_f2p_collection_error: bool = False
    # v6c: True after the post-edit pytest nudge has fired once. The
    # nudge surfaces the exact ``pytest <id1> <id2> ...`` invocation
    # the moment the first successful edit lands in bench/yolo mode,
    # so the model doesn't need to wait for a finish_plan intercept
    # to learn what to run.
    post_edit_pytest_nudge_sent: bool = False


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

def is_test_command(cmd: str) -> bool:
    """Return True if ``cmd`` looks like a test invocation (not just a path containing 'test').

    Requires an actual test runner keyword at the start of the command or after
    a shell separator.  Simply having 'test_' in a file path (e.g. ``ls tests/``)
    does not count.
    """
    # Strip leading cd/env prefix to find the actual command.
    text = cmd.strip()
    while text.startswith("cd "):
        for sep in ("&&", ";"):
            idx = text.find(sep)
            if idx != -1:
                text = text[idx + len(sep):].strip()
                break
        else:
            break
    # Check if the command starts with a real test runner or runs a test script.
    _RUNNERS = ("pytest", "python -m pytest", "python -m unittest", "unittest",
                "python test_", "python3 test_", "python ./test_", "python3 ./test_")
    text_lower = text.lower()
    return any(text_lower.startswith(r) or f" {r}" in text_lower for r in _RUNNERS)


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
        if "::" not in tid:
            continue
        path = tid.split("::", 1)[0].replace("\\", "/").strip()
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
    cmd_norm = cmd.replace("\\", "/")
    cmd_lower = cmd_norm.lower()

    # Bare pytest invocation (no positional path arg) — assume full suite.
    # Heuristic: pytest is mentioned but no F2P test path appears as a
    # substring AND no .py path appears anywhere on the line.
    has_runner = ("pytest" in cmd_lower) or ("unittest" in cmd_lower)
    has_any_path = ".py" in cmd_norm or "::" in cmd_norm
    if has_runner and not has_any_path:
        return set(files)

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


