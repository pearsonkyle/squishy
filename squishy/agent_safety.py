"""Quality gates, informational nudges, and test-failure feedback for the agent loop."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from squishy.agent_state import LoopState, TaskResult
from squishy.quality import assess_response, build_correction
from squishy.tool_restrictions import profile_has_edit_tool
from squishy.tools import REGISTRY

if TYPE_CHECKING:
    from squishy.agent import Agent
    from squishy.client import ToolCall


def can_nudge(st: LoopState, turn: int, min_gap: int = 2, max_total: int = 12) -> bool:
    """Return True if enough turns have passed since the last nudge and
    the total nudge count hasn't exceeded the cap."""
    if st.total_nudges >= max_total:
        return False
    return turn - st.last_nudge_turn >= min_gap


def record_nudge(st: LoopState, turn: int) -> None:
    """Mark that a nudge was injected this turn."""
    if turn == st.last_nudge_turn:
        st.nudges_this_turn += 1
    else:
        st.nudges_this_turn = 1
    st.last_nudge_turn = turn
    st.total_nudges += 1


# Even control-flow nudges (force=True, min_gap=0) are capped per turn.
# Non-forced nudges are already limited to one per turn by ``min_gap``; without
# this, several forced gates could each append a [system] message in the same
# turn, spending context and (on strict-alternation templates) stacking
# consecutive user turns.
MAX_NUDGES_PER_TURN = 2


def inject_nudge(
    agent: Agent, st: LoopState, turn: int, content: str,
    *, min_gap: int = 2, force: bool = False,
) -> bool:
    """Inject a system nudge if under the cap. Returns True if injected.

    Mirrors the nudge to ``agent.display.nudge`` so the user can see
    every correction the harness sends to the model — closing the
    long-standing opacity gap where agent behavior changes silently
    in response to invisible system messages.
    """
    # Hard ceiling: even forced nudges stop after 2x the normal cap.
    hard_cap = agent.config.max_system_nudges * 2
    if st.total_nudges >= hard_cap:
        return False
    # Per-turn ceiling applies to forced nudges too (see MAX_NUDGES_PER_TURN).
    if turn == st.last_nudge_turn and st.nudges_this_turn >= MAX_NUDGES_PER_TURN:
        return False
    if not force and not can_nudge(st, turn, min_gap=min_gap):
        return False
    agent.messages.append({"role": "user", "content": content})
    record_nudge(st, turn)
    if agent.display is not None:
        agent.display.nudge(content)
    return True


def needs_f2p_verification(st: LoopState) -> bool:
    """True iff the agent edited but hasn't run F2P tests since that edit.

    Used by the v2 auto-pytest finish gate (agent.py natural-finish and
    done-phase exit) to decide whether to synthesize a pytest run before
    accepting the agent's "I'm done" claim.
    """
    return bool(st.files_edited) and st.last_edit_turn > st.last_f2p_test_turn


def apply_quality_gate(
    agent: Agent, tool_calls: list[ToolCall], st: LoopState, turn: int,
) -> TaskResult | str | None:
    """Quality gate: catch degenerate tool calls before dispatching.

    Returns:
      TaskResult — forced finish (end the run)
      "skip"    — quality issue detected, correction injected, skip dispatch
      None      — proceed normally
    """
    _is_constrained = agent.config.permission_mode in ("bench", "yolo")

    ok, reason = assess_response(
        tool_calls, agent.messages, REGISTRY,
        edit_fail_paths=frozenset(st.recent_edit_fail_files),
    )
    if ok:
        st.quality_retries = 0
        return None

    st.total_quality_violations += 1

    loop_reasons = ("repeated_tool_call", "excessive_reread", "repeated_command",
                    "edit_verify_loop", "repeated_recall", "repeated_search",
                    "plan_loop")

    if _is_constrained:
        if reason in loop_reasons:
            # Raised from 4→6 (post-cooldown).  With cooldown of -2 per
            # successful edit, this is effectively ~3 strikes after the
            # last edit, instead of accumulating from exploration-phase
            # quality noise.
            if st.files_edited and (st.total_quality_violations >= 6 or st.test_passed_after_edit):
                if agent.display:
                    agent.display.warn(
                        f"quality: {reason} — force finishing (violations={st.total_quality_violations})"
                    )
                return agent._build_result(
                    st, success=True,
                    final_text="Fix applied. The agent completed edits but entered a loop.",
                    turn=turn,
                )
            if not st.files_edited and st.total_quality_violations >= 15:
                if agent.display:
                    agent.display.warn(
                        f"quality: {reason} — force finishing with no edits "
                        f"(violations={st.total_quality_violations})"
                    )
                return agent._build_result(
                    st, success=False,
                    error=f"quality loop: {st.total_quality_violations} violations without edits",
                    turn=turn,
                )

    if st.quality_retries < agent.config.max_quality_retries:
        st.quality_retries += 1
        correction = build_correction(reason)
        if agent.display:
            agent.display.warn(f"quality: {reason}")
        if st.total_nudges < agent.config.max_system_nudges * 2:
            full_content = f"[system] {correction}"
            agent.messages.append(
                {"role": "user", "content": full_content}
            )
            record_nudge(st, turn)
            if agent.display is not None:
                agent.display.nudge(full_content)
        st.quality_skips += 1
        return "skip"

    if _is_constrained and reason in loop_reasons and st.files_edited:
        if agent.display:
            agent.display.warn(f"quality: {reason} after edits — finishing")
        return agent._build_result(
            st, success=True,
            final_text="Fix applied. The agent completed edits but entered a loop.",
            turn=turn,
        )

    if agent.display:
        agent.display.warn(f"quality: {reason} (retries exhausted, proceeding)")
    st.quality_retries = 0
    return None


def inject_test_failure_nudge(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    *, turn: int = 0,
) -> None:
    """After a test failure, inject structured feedback with specific failures.

    Fires on every failed test run (not just after 2+ cycles) so the model
    always gets structured guidance about what to fix.  Escalates at 4+ cycles.
    """
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    if tc.name != "run_command":
        return
    data = outcome.get("data", {})
    exit_code = data.get("exit_code", 0)
    if exit_code == 0:
        return
    command = str(tc.args.get("command", ""))
    if not any(kw in command for kw in ("pytest", "test", "unittest")):
        return

    # Pytest exit code 5 = "no tests collected".  This means the test doesn't
    # exist yet or the path is wrong — pressuring edit_file is counterproductive.
    # Similarly, exit code 4 = "usage error" (bad args).
    # Exit code 2 with ImportError/ModuleNotFoundError = environment issue.
    stdout = str(data.get("stdout", ""))
    stderr = str(data.get("stderr", ""))
    combined_output = stdout + stderr
    combined_lower = combined_output.lower()

    # Read F2P + install_status hints from the bench harness (threaded via
    # ToolContext.notes — see swebench.run_swebench_instance).
    notes = getattr(agent.tool_ctx, "notes", {}) or {}
    fail_to_pass: list[str] = []
    raw_f2p = notes.get("fail_to_pass_tests")
    if raw_f2p:
        try:
            parsed = json.loads(raw_f2p)
            if isinstance(parsed, list):
                fail_to_pass = [str(t) for t in parsed]
        except (ValueError, TypeError):
            pass
    install_ok = True
    raw_install = notes.get("install_status")
    if raw_install:
        try:
            parsed_i = json.loads(raw_install)
            if isinstance(parsed_i, dict):
                install_ok = bool(parsed_i.get("ok", True))
        except (ValueError, TypeError):
            pass

    # Detect import/environment errors separately from missing tests.
    import_error = (
        "importerror" in combined_lower
        or "modulenotfounderror" in combined_lower
        or "no module named" in combined_lower
    )
    if import_error and exit_code in (1, 2):
        # When install_deps already failed, the import error is an environment
        # artifact — do not pressure the agent to "fix imports".  Instead,
        # remind it the source edit is what matters.
        if not install_ok:
            inject_nudge(agent, st, turn, (
                "[system] ImportError is a broken test environment, not your bug. "
                "Do not fix imports or install packages — edit the source to fix "
                "the reported bug."
            ), min_gap=2)
            return
        inject_nudge(agent, st, turn, (
            "[system] ImportError is an environment artifact, not your bug. Edit "
            "the source to fix the reported bug; to re-test use "
            "`python -m pytest path/to/test.py::specific_test -x`."
        ), min_gap=2)
        return

    no_tests_collected = (
        exit_code == 5
        or "no tests ran" in combined_lower
        or "collected 0 items" in combined_lower
        or "no tests were selected" in combined_lower
    )
    if no_tests_collected:
        is_bench = agent.config.permission_mode == "bench"
        if is_bench:
            inject_nudge(agent, st, turn, (
                "[system] No tests collected — the test is added by the harness "
                "after your fix. Run the whole file (`python -m pytest "
                "path/to/test.py -x`) and fix the source to match the behavior the "
                "failing test name describes."
            ), min_gap=2)
        else:
            inject_nudge(agent, st, turn, (
                "[system] No tests collected — the path or test name is likely "
                "wrong. Use `search_files` to locate the right test file."
            ), min_gap=2)
        return

    # Build structured failure summary from parsed test output.
    test_summary = data.get("test_summary")
    failure_lines = ""
    f2p_header_extra = ""
    if test_summary:
        failures = test_summary.get("failures", [])
        passed = test_summary.get("passed", 0)
        failed = test_summary.get("failed", 0)
        errors = test_summary.get("errors", 0)
        header = f"Test results: {passed} passed, {failed} failed"
        if errors:
            header += f", {errors} errors"
        header += "."

        # F2P-aware ordering: pull eval-target failures to the top, count
        # how many of the F2P set are still failing so the agent sees the
        # signal that actually determines pass/fail.
        f2p_count_now = -1  # -1 means "no F2P info available"
        if failures and fail_to_pass:
            def _is_f2p(f: dict) -> bool:
                tname = str(f.get("test", ""))
                return any(
                    f2p in tname or tname in f2p
                    for f2p in fail_to_pass
                )

            f2p_failing = [f for f in failures if _is_f2p(f)]
            other_failing = [f for f in failures if not _is_f2p(f)]
            failures = f2p_failing + other_failing
            f2p_count_now = len(f2p_failing)
            if f2p_failing:
                f2p_header_extra = (
                    f"\n⚠ {f2p_count_now}/{len(fail_to_pass)} of the FAIL_TO_PASS "
                    f"evaluation tests are still failing — these are what the "
                    f"benchmark scores on."
                )
                # Cross-cycle F2P-only progress: warn when F2P failure count
                # has not decreased.  An agent fixing unrelated tests but
                # leaving F2P red still gets the generic "Progress!" reward
                # — the F2P-specific signal counters that.
                if st.last_f2p_failure_count >= 0:
                    prev_f2p = st.last_f2p_failure_count
                    if f2p_count_now >= prev_f2p:
                        f2p_header_extra += (
                            f"\n⚠ F2P failure count did not decrease this cycle "
                            f"({prev_f2p} → {f2p_count_now}). "
                            "Your last edit did not improve the EVAL TARGET — "
                            "non-F2P fixes do not count."
                        )
                    else:
                        green = len(fail_to_pass) - f2p_count_now
                        f2p_header_extra += (
                            f"\nF2P progress: {green}/{len(fail_to_pass)} target tests now pass."
                        )
        # Always track F2P count across cycles, even when 0 failing.
        if fail_to_pass and f2p_count_now < 0:
            f2p_count_now = 0
        if f2p_count_now >= 0:
            st.last_f2p_failure_count = f2p_count_now

        if failures:
            items = []
            for i, f in enumerate(failures[:5], 1):
                items.append(f"  {i}. {f['test']} — {f['error']}" if f['error'] else f"  {i}. {f['test']}")
            failure_lines = "\nFailed tests:\n" + "\n".join(items)

        # Cross-cycle comparison.
        comparison = ""
        cur_count = failed + errors
        cur_names = [f["test"] for f in failures]
        if st.last_test_failure_count >= 0:
            prev = st.last_test_failure_count
            if cur_count < prev:
                comparison = f"\nProgress: failures decreased from {prev} to {cur_count}. Keep iterating."
            elif cur_count > prev:
                comparison = f"\nWARNING: Your last edit INCREASED failures from {prev} to {cur_count}. Consider reverting."
            else:
                prev_names = set(st.last_test_failures)
                cur_names_set = set(cur_names)
                if prev_names == cur_names_set:
                    comparison = "\nSame tests still failing. Your edit had no effect. Try a DIFFERENT approach."
                else:
                    comparison = "\nYou fixed some tests but broke others. Check the newly failing tests."

        st.last_test_failure_count = cur_count
        st.last_test_failures = cur_names
    else:
        header = "Test failed."
        failure_lines = ""
        comparison = ""

    # Structured feedback — always informational, never threatening.
    if f2p_header_extra and failure_lines:
        action = (
            "Focus your next `edit_file` on the FIRST FAIL_TO_PASS test above — "
            "non-F2P failures are noise."
        )
    elif failure_lines:
        action = "Focus your next `edit_file` on fixing the FIRST failing test."
    else:
        action = "Call `edit_file` with your fix now."
    inject_nudge(agent, st, turn, (
        f"[system] {header}{f2p_header_extra}{failure_lines}{comparison}\n{action}"
    ), min_gap=2)


def check_goal_drift(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    *, turn: int = 0,
) -> None:
    """Detect when the agent is fixing environmental issues instead of the bug."""
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    from squishy.agent_state import path_matches_problem

    if tc.name == "edit_file" and outcome.get("success"):
        path = str(tc.args.get("path", ""))
        if st.problem_files and not path_matches_problem(path, st.problem_files):
            st.env_fix_files.add(path)

    if tc.name == "run_command" and not outcome.get("success"):
        data = outcome.get("data", {})
        output = str(data.get("stderr", "")) + str(data.get("stdout", ""))
        env_patterns = (
            "ImportError", "ModuleNotFoundError", "No module named",
            "cannot import name", "collections.Mapping",
            "collections.MutableMapping", "collections.Callable",
        )
        if any(p in output for p in env_patterns):
            st.env_error_count += 1

    should_nudge = (
        (len(st.env_fix_files) >= 2 and st.env_error_count >= 2)
        or st.env_error_count >= 3
    )
    if should_nudge:
        st.env_error_count = 0
        st.env_fix_files.clear()
        inject_nudge(agent, st, turn, (
            "[system] You are fixing environment/import errors, not the reported "
            "bug. Those are test-environment artifacts. Edit the source described "
            "in the problem statement instead."
        ), min_gap=2, force=True)


def track_edit_failure(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    *, turn: int = 0,
) -> TaskResult | None:
    """Track edit_file failures per file and nudge on repeated failures.

    Returns a TaskResult to force-finish when the agent retries the SAME
    `old_str` against the SAME path 5+ times in a row — a loop the existing
    repeated_tool_call quality gate misses because `new_str` typically varies.
    """
    if agent.config.permission_mode not in ("bench", "yolo"):
        return None
    if tc.name != "edit_file":
        return None

    path = str(tc.args.get("path", ""))
    if not path:
        return None

    if not outcome.get("success"):
        st.edit_failures_per_file[path] = st.edit_failures_per_file.get(path, 0) + 1
        st.total_edit_failures += 1
        failures = st.edit_failures_per_file[path]

        # Track identical-old_str repeats per path. Hash for compactness.
        old_str = str(tc.args.get("old_str", ""))
        old_hash = hashlib.blake2b(old_str.encode("utf-8", "replace"), digest_size=8).hexdigest()
        prev = st.last_edit_old_str_per_file.get(path)
        if prev and prev[0] == old_hash:
            identical_count = prev[1] + 1
        else:
            identical_count = 1
        st.last_edit_old_str_per_file[path] = (old_hash, identical_count)

        # v6d: targeted escape hatch for "old_str not found" loops. Fire at
        # count >= 2 (one turn before the generic 3-failure message) with
        # action-specific guidance: re-read the file before guessing again.
        # The unleash-157 case in v6c showed an agent burning 3 turns on
        # successive variants of the same broken old_str without ever
        # re-reading. Targeting only the "not found" error keeps this
        # complementary to the generic message at >=3 (which covers
        # permission errors, write_file outcomes, etc).
        err_text = str(outcome.get("error") or "")
        if failures >= 2 and "old_str not found" in err_text:
            inject_nudge(agent, st, turn, (
                f"[system] old_str still doesn't match `{path}`. Stop guessing: "
                f"call read_file(path=\"{path}\") and copy the exact lines "
                f"(including whitespace) into old_str."
            ), min_gap=1)

        # Force-finish after 5 identical-old_str failures to the same path.
        # The agent is in a loop the model can't escape; the prior fs.py
        # Stage 1c fallback masks most cases but not all.
        if identical_count >= 5:
            if agent.display:
                agent.display.warn(
                    f"edit-loop: {identical_count}× identical old_str failures to `{path}` — force finishing"
                )
            if st.files_edited:
                return agent._build_result(
                    st, success=True,
                    final_text=(
                        f"Edit loop detected: {identical_count} identical-old_str failures "
                        f"to `{path}`. Prior edits applied; ending the run."
                    ),
                    turn=turn,
                )
            return agent._build_result(
                st, success=False,
                error=(
                    f"edit loop: {identical_count} identical-old_str failures to "
                    f"`{path}` with no successful edits"
                ),
                turn=turn,
            )

        if failures >= 3:
            inject_nudge(agent, st, turn, (
                f"[system] {failures} failed edits to `{path}`. Read the exact "
                "line range, then copy that text verbatim into old_str with 2-3 "
                "lines of surrounding context."
            ), min_gap=2)
    else:
        st.edit_failures_per_file[path] = 0
        st.last_edit_old_str_per_file.pop(path, None)
    return None


_FILE_READ_CMD = re.compile(
    r"^\s*(?:sed\s+-n|cat\s|head\s|tail\s|awk\s)", re.IGNORECASE,
)


def detect_shell_file_read(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    *, turn: int = 0,
) -> None:
    """Nudge when the model uses shell commands to read files instead of read_file."""
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    # Pointless — and actively misleading — when there is no read_file to point
    # at: under a shell-only profile `cat`/`sed` is the intended way to read.
    if not profile_has_edit_tool(agent.config.tool_profile):
        return
    if tc.name != "run_command":
        return
    command = str(tc.args.get("command", ""))
    if _FILE_READ_CMD.search(command):
        inject_nudge(agent, st, turn, (
            "[system] Use the `read_file` tool to read files; keep `run_command` "
            "for tests, builds, and git."
        ), min_gap=3)


