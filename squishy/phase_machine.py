"""Phase-gated state machine for bench mode agent behavior.

Each phase defines:
  - A set of tool names the LLM can see in the schema
  - Transition conditions (checked after each tool dispatch)
  - An optional turn budget

Phases: explore -> plan -> execute -> verify <-> execute -> done

This replaces the nudge-heavy approach with structural control:
instead of threatening the model to stop exploring, we simply
remove exploration tools from the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from squishy.client import ToolCall

from dataclasses import field

from squishy.agent_state import (
    distinct_f2p_files,
    f2p_files_in_command,
    is_test_command,
)

# -- Tool sets per phase ----------------------------------------------------

EXPLORE_PHASE_TOOLS = frozenset({
    "read_file", "list_directory", "search_files", "glob_files",
    "recall", "save_note", "show_diff", "run_command",
})

PLAN_PHASE_TOOLS = frozenset({
    "plan_task", "save_note", "recall",
})

EXECUTE_PHASE_TOOLS = frozenset({
    "read_file", "edit_file", "write_file", "undo_edit",
    "run_command", "update_plan", "save_note", "show_diff", "get_plan",
})

VERIFY_PHASE_TOOLS = frozenset({
    "run_command", "show_diff", "finish_plan", "update_plan",
    "get_plan", "read_file",
})

DONE_PHASE_TOOLS: frozenset[str] = frozenset()

PHASE_TOOLS: dict[str, frozenset[str]] = {
    "explore": EXPLORE_PHASE_TOOLS,
    "plan": PLAN_PHASE_TOOLS,
    "execute": EXECUTE_PHASE_TOOLS,
    "verify": VERIFY_PHASE_TOOLS,
    "done": DONE_PHASE_TOOLS,
}


def tools_for_phase(phase: str) -> frozenset[str]:
    """Return the set of tool names available in the given phase."""
    return PHASE_TOOLS.get(phase, DONE_PHASE_TOOLS)


# -- State and transition types ---------------------------------------------

@dataclass
class PhaseState:
    """Mutable state for the phase machine."""
    phase: str = "explore"
    explore_turns: int = 0
    plan_turns: int = 0
    execute_turns_since_edit: int = 0
    fix_verify_cycles: int = 0
    has_recall: bool = False
    has_edit: bool = False
    has_test_run: bool = False
    test_passed_after_edit: bool = False

    # Budgets (set from Config at init).
    max_explore_turns: int = 8
    max_plan_turns: int = 3
    max_fix_verify_cycles: int = 6

    # F5: F2P file-coverage tracking.  Set from harness via
    # ``fail_to_pass``; ``test_passed_after_edit`` only flips True when
    # ``f2p_files_covered`` is a superset of ``distinct_f2p_files(...)``.
    fail_to_pass: list[str] = field(default_factory=list)
    f2p_files_covered: set[str] = field(default_factory=set)


@dataclass
class Transition:
    """Result of checking for a phase transition."""
    new_phase: str | None = None
    notification: str = ""
    force_finish: bool = False
    force_finish_success: bool = False


# -- Transition logic -------------------------------------------------------

def check_transition(
    ps: PhaseState,
    dispatched: list[tuple[ToolCall, dict[str, Any]]],
) -> Transition:
    """Check if dispatched tool calls trigger a phase transition.

    Call this after every turn's tool dispatch. Returns a Transition
    describing what happened (or no transition if nothing changed).
    """
    # Analyze what happened this turn.
    had_recall = False
    had_plan_success = False
    had_edit_success = False
    had_test_run = False
    had_test_pass = False
    had_finish_plan = False

    for tc, outcome in dispatched:
        if tc.name == "recall":
            had_recall = True
        elif tc.name == "plan_task" and outcome.get("success"):
            had_plan_success = True
        elif tc.name == "edit_file" and outcome.get("success"):
            had_edit_success = True
        elif tc.name == "run_command":
            cmd = str(tc.args.get("command", ""))
            if is_test_command(cmd):
                had_test_run = True
                data = outcome.get("data", {})
                if data.get("exit_code") == 0:
                    had_test_pass = True
                    # F5: record which F2P files this command covered, but
                    # only when structured parsing didn't surface failures
                    # (a false-positive exit 0 must not count as coverage).
                    test_summary = data.get("test_summary") or {}
                    if not test_summary.get("failed", 0):
                        covered = f2p_files_in_command(cmd, ps.fail_to_pass)
                        ps.f2p_files_covered.update(covered)
        elif tc.name == "finish_plan" and outcome.get("success"):
            had_finish_plan = True

    # Update cumulative flags.
    if had_recall:
        ps.has_recall = True
    if had_edit_success:
        ps.has_edit = True
    if had_test_run:
        ps.has_test_run = True
    if had_test_pass and ps.has_edit:
        # F5: only declare "tests pass after edit" when *every* distinct
        # F2P file has been exercised by a passing test command.  When
        # the harness didn't supply F2P (interactive use, terminal-bench,
        # etc.) ``needed`` is empty and we keep the legacy behavior.
        needed = distinct_f2p_files(ps.fail_to_pass)
        if not needed or ps.f2p_files_covered >= needed:
            ps.test_passed_after_edit = True

    # -- Phase: explore --
    if ps.phase == "explore":
        ps.explore_turns += 1
        # If the model called plan_task early (skipping explore), jump to execute.
        if had_plan_success:
            return Transition(
                new_phase="execute",
                notification=(
                    "[system] Phase: execute. Plan approved (early). "
                    "Read the target file, then call `edit_file` to apply your fix. "
                    "Call `update_plan` after each step is done."
                ),
            )
        if ps.has_recall or ps.explore_turns >= ps.max_explore_turns:
            return Transition(
                new_phase="plan",
                notification=(
                    "[system] Phase: plan. "
                    "Call `plan_task` with your fix strategy (problem, solution, steps). "
                    "The plan is auto-approved in bench mode."
                ),
            )

    # -- Phase: plan --
    elif ps.phase == "plan":
        ps.plan_turns += 1
        if had_plan_success:
            return Transition(
                new_phase="execute",
                notification=(
                    "[system] Phase: execute. Plan approved. "
                    "Read the target file, then call `edit_file` to apply your fix. "
                    "Call `update_plan` after each step is done."
                ),
            )
        if ps.plan_turns >= ps.max_plan_turns:
            return Transition(
                new_phase="execute",
                notification=(
                    "[system] Phase: execute. Plan budget exhausted — proceeding without formal plan. "
                    "Read the target file and apply your fix directly."
                ),
            )

    # -- Phase: execute --
    elif ps.phase == "execute":
        if ps.has_edit and ps.has_test_run:
            # If test already passed, skip verify and go straight to done.
            if ps.test_passed_after_edit:
                return Transition(
                    new_phase="done",
                    notification=(
                        "[system] Phase: done. Tests passed after edit. "
                        "Respond with a summary of what you changed and why."
                    ),
                )
            return Transition(
                new_phase="verify",
                notification=(
                    "[system] Phase: verify. "
                    "Check your test results. If tests pass, call `finish_plan`. "
                    "If tests fail, you will return to execute phase."
                ),
            )
        # Track turns since first edit without a test run.
        if ps.has_edit and not ps.has_test_run:
            ps.execute_turns_since_edit += 1
            if ps.execute_turns_since_edit >= 2:
                return Transition(
                    notification=(
                        "[system] You have edited code but have NOT run the failing "
                        "tests yet. Call `run_command` with the test command NOW. "
                        "Do not call `update_plan` or `finish_plan` until tests pass."
                    ),
                )

    # -- Phase: verify --
    elif ps.phase == "verify":
        if had_finish_plan or ps.test_passed_after_edit:
            return Transition(
                new_phase="done",
                notification=(
                    "[system] Phase: done. "
                    "Respond with a summary of what you changed and why."
                ),
            )
        if had_test_run and not had_test_pass:
            ps.fix_verify_cycles += 1
            if ps.fix_verify_cycles >= ps.max_fix_verify_cycles:
                return Transition(
                    force_finish=True,
                    force_finish_success=ps.has_edit,
                )
            # Reset for next execute->verify cycle.
            ps.has_edit = False
            ps.has_test_run = False
            ps.test_passed_after_edit = False
            return Transition(
                new_phase="execute",
                notification=(
                    f"[system] Phase: execute. Test failed "
                    f"(cycle {ps.fix_verify_cycles}/{ps.max_fix_verify_cycles}). "
                    "Read the error, fix the code, and run tests again."
                ),
            )

    return Transition()  # No transition.


def advance(ps: PhaseState, transition: Transition) -> None:
    """Apply a transition to the phase state."""
    if transition.new_phase:
        ps.phase = transition.new_phase


def check_finish_plan_gate(
    ps: PhaseState,
    dispatched: list[tuple[ToolCall, dict[str, Any]]],
    fail_to_pass: list[str],
    intercepts_so_far: int,
    *,
    max_intercepts: int = 1,
    last_f2p_failures: list[dict[str, str]] | None = None,
    last_f2p_collection_error: bool = False,
) -> str:
    """Intercept `finish_plan` when the agent declares done without proof
    that the FAIL_TO_PASS tests actually pass.

    Returns a nudge string when the call should be blocked.  Returns ""
    when finish_plan is allowed through (either because the F2P tests
    have demonstrably passed, or the bypass threshold has been reached
    so the agent isn't trapped by a broken test environment).

    Args:
        ps: phase state (used to confirm we're in verify phase).
        dispatched: this turn's tool calls (we look for finish_plan).
        fail_to_pass: F2P test identifiers from the bench harness.
        intercepts_so_far: how many times we have already blocked.
        max_intercepts: release the gate after this many intercepts.
        last_f2p_failures: v5 — failing F2P test details from the most
            recent pytest run (``[{"test": nodeid, "error": msg}, ...]``).
            When non-empty the gate emits a partial-pass message that
            enumerates the still-failing tests by ID + assertion.
        last_f2p_collection_error: v6b — True when the most recent F2P
            run reported pytest errors (collection / import failures)
            but no per-test failures.  When set and ``last_f2p_failures``
            is empty, the gate emits a "fix collection first" hint.
            Ignored when ``last_f2p_failures`` is non-empty (per-test
            failures are strictly more actionable).
    """
    # Only enforce in verify phase.  Earlier finish_plan calls are
    # already prevented by phase tool restrictions.
    if ps.phase != "verify":
        return ""
    # Nothing to enforce when the harness didn't supply F2P tests.
    if not fail_to_pass:
        return ""
    # Bypass after enough intercepts.  Some instances can't run their
    # tests at all (degraded install_status) — don't trap the agent.
    if intercepts_so_far >= max_intercepts:
        return ""
    # If the F2P tests demonstrably passed, allow the finish.
    if ps.test_passed_after_edit:
        return ""
    # Look for a finish_plan call this turn.
    for tc, _outcome in dispatched:
        if tc.name == "finish_plan":
            # v5: partial-pass branch — agent's last F2P run reported
            # failures.  Enumerate the still-failing tests so the model
            # can target a focused fix rather than guessing what to
            # change.  Takes priority over the F5 file-coverage message
            # because per-test failures are strictly more actionable.
            if last_f2p_failures:
                lines: list[str] = []
                for f in last_f2p_failures[:5]:
                    test_id = f.get("test", "")
                    err = f.get("error", "")
                    line = f"  - `{test_id}`"
                    if err:
                        line += f"\n      → {err}"
                    lines.append(line)
                more_msg = (
                    f"\n  ...and {len(last_f2p_failures) - 5} more"
                    if len(last_f2p_failures) > 5 else ""
                )
                return (
                    "[system] Held finish_plan: your most recent F2P test "
                    "run reported failures.  These FAIL_TO_PASS tests are "
                    "still failing — fix them before finishing:\n"
                    + "\n".join(lines) + more_msg +
                    "\n\nEither edit_file the source to address the "
                    "assertion(s), then re-run pytest on these specific "
                    "test IDs to verify, or call "
                    "finish_plan(status=\"failure\") if the bug is "
                    "genuinely unfixable."
                )
            # v6b: collection-error branch — pytest reported errors
            # (collection / import) but produced no per-test failures
            # to enumerate.  Tell the agent to fix the collection
            # error first; the F2P tests have not actually run.
            if last_f2p_collection_error:
                return (
                    "[system] Held finish_plan: your most recent F2P "
                    "test run crashed at collection (import / missing "
                    "module / syntax error) — no F2P tests actually "
                    "ran.  Fix the collection error first (typically "
                    "a missing import or stale dependency in the test "
                    "file or its imports), then re-run pytest on the "
                    "specific F2P test IDs to verify, before calling "
                    "finish_plan again."
                )
            # F5: if the agent ran SOME but not all F2P files, the gate
            # message should call out the missing ones explicitly — the
            # generic "run the failing tests" wording lets the agent
            # think they're done after one file passes (the v27.1
            # scico-561 failure mode).
            needed = distinct_f2p_files(fail_to_pass)
            missing = sorted(needed - ps.f2p_files_covered)
            if needed and ps.f2p_files_covered and missing:
                miss_preview = ", ".join(f"`{p}`" for p in missing[:3])
                more = f" (and {len(missing) - 3} more)" if len(missing) > 3 else ""
                return (
                    "[system] Held finish_plan: you ran "
                    f"{len(ps.f2p_files_covered)}/{len(needed)} of the "
                    "FAIL_TO_PASS test files successfully, but these "
                    f"file(s) have NOT been exercised yet: {miss_preview}"
                    f"{more}.  The same bug class may be present in those "
                    "files too — read them, apply the analogous fix, and "
                    "run them.  If the missing tests are unrunnable in "
                    "this environment, call finish_plan once more to bypass."
                )
            preview = ", ".join(f"`{t}`" for t in fail_to_pass[:3])
            more = f" (and {len(fail_to_pass) - 3} more)" if len(fail_to_pass) > 3 else ""
            return (
                "[system] Held finish_plan: you have not actually run the "
                "FAIL_TO_PASS tests successfully yet. Before declaring done, "
                f"run those specific tests: `pytest {' '.join(fail_to_pass[:3])}` "
                f"({preview}{more}). If they pass, call finish_plan again. "
                "If the test environment is broken, call finish_plan once more "
                "to bypass this check."
            )
    return ""
