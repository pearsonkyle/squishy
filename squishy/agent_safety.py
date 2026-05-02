"""Loop detection, quality gates, nudges, and stuck detection for the agent loop."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from squishy.agent_state import LoopState, TaskResult
from squishy.quality import assess_response, build_correction
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
    st.last_nudge_turn = turn
    st.total_nudges += 1


def inject_nudge(
    agent: Agent, st: LoopState, turn: int, content: str,
    *, min_gap: int = 2, force: bool = False,
) -> bool:
    """Inject a system nudge if under the cap. Returns True if injected."""
    # Hard ceiling: even forced nudges stop after 2x the normal cap.
    hard_cap = agent.config.max_system_nudges * 2
    if st.total_nudges >= hard_cap:
        return False
    if not force and not can_nudge(st, turn, min_gap=min_gap):
        return False
    agent.messages.append({"role": "user", "content": content})
    record_nudge(st, turn)
    return True


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

    ok, reason = assess_response(tool_calls, agent.messages, REGISTRY)
    if ok:
        st.quality_retries = 0
        return None

    st.total_quality_violations += 1

    loop_reasons = ("repeated_tool_call", "excessive_reread", "repeated_command",
                    "edit_verify_loop", "repeated_recall", "repeated_search")

    if _is_constrained:
        if reason in loop_reasons:
            if st.files_edited and (st.total_quality_violations >= 4 or st.test_passed_after_edit):
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
            agent.messages.append(
                {"role": "user", "content": f"[system] {correction}"}
            )
            record_nudge(st, turn)
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


def apply_stuck_detection(
    agent: Agent, st: LoopState, is_bench: bool, turn: int = 0,
) -> None:
    """Track file-mutation progress and inject stuck nudges in bench mode."""
    made_progress = (
        len(st.files_created) > st.prior_created_len
        or len(st.files_edited) > st.prior_edited_len
    )
    if made_progress:
        st.turns_without_progress = 0
    else:
        st.turns_without_progress += 1
    st.prior_created_len = len(st.files_created)
    st.prior_edited_len = len(st.files_edited)

    if (
        is_bench
        and st.turns_without_progress >= agent.config.max_stuck_turns
        and st.turns_without_progress % agent.config.max_stuck_turns == 0
    ):
        urgency = (
            "URGENT" if st.turns_without_progress >= agent.config.max_stuck_turns * 2
            else "WARNING"
        )
        wd = agent.tool_ctx.working_dir
        already_read = sorted(
            os.path.relpath(p, wd) if os.path.isabs(p) else p
            for p in agent.tool_ctx.files_read_count.keys()
        )[:5]
        already_note = (
            f"\nYou have already read: {', '.join(already_read)}. "
            "Do NOT read these files again — use the content you already have."
        ) if already_read else ""

        if not already_read and st.problem_files:
            hint_files = sorted(st.problem_files)[:3]
            hint = f"\nHint: the problem mentions: {', '.join(hint_files)}. Try reading one of those."
        else:
            hint = ""

        inject_nudge(agent, st, turn, (
            f"[system] [{urgency}] You have NOT edited any files in "
            f"{st.turns_without_progress} turns. You MUST call `edit_file` NOW.\n"
            "Stop reading, searching, and exploring. You have enough information.\n"
            "1. Pick the most likely file and function from the problem statement.\n"
            "2. Call `edit_file` with your best fix attempt using content you already have.\n"
            "A wrong fix that you iterate on is better than more exploration."
            f"{already_note}{hint}"
        ))


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
    if data.get("exit_code", 0) == 0:
        return
    command = str(tc.args.get("command", ""))
    if not any(kw in command for kw in ("pytest", "test", "unittest")):
        return

    # Build structured failure summary from parsed test output.
    test_summary = data.get("test_summary")
    failure_lines = ""
    if test_summary:
        failures = test_summary.get("failures", [])
        passed = test_summary.get("passed", 0)
        failed = test_summary.get("failed", 0)
        errors = test_summary.get("errors", 0)
        header = f"Test results: {passed} passed, {failed} failed"
        if errors:
            header += f", {errors} errors"
        header += "."

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

    # Escalation based on cycle count.
    if st.fix_verify_cycles >= 4:
        inject_nudge(agent, st, turn, (
            f"[system] CRITICAL: {st.fix_verify_cycles} fix-verify cycles "
            f"and the test still fails. {header}{failure_lines}{comparison}\n\n"
            "STOP making small tweaks — your approach is fundamentally wrong. "
            "You MUST:\n"
            "1. Re-read the failing test to understand exactly what it expects.\n"
            "2. Re-read the problem statement to check your understanding.\n"
            "3. Try a COMPLETELY different fix strategy.\n"
            "If you cannot fix it, respond with a text summary and stop."
        ), min_gap=0)
    else:
        action = (
            "Focus your next `edit_file` on fixing the FIRST failing test."
            if failure_lines else
            "Call `edit_file` with your fix now."
        )
        inject_nudge(agent, st, turn, (
            f"[system] {header}{failure_lines}{comparison}\n{action}"
        ), min_gap=0)


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
            "[system] GOAL DRIFT WARNING: You are fixing environmental/import "
            "errors instead of the actual bug described in the problem statement. "
            "These import errors are caused by Python version differences in the "
            "test environment — they are NOT the bug you need to fix.\n\n"
            "STOP fixing import compatibility issues. Instead:\n"
            "1. Focus ONLY on the bug described in the problem statement.\n"
            "2. Run a more targeted test: "
            "`python -m pytest path/to/test.py::specific_test -x`\n"
            "3. Or write a minimal reproduction script to verify your fix.\n\n"
            "Re-read the problem statement and get back on track."
        ), min_gap=0)


def track_edit_failure(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    *, turn: int = 0,
) -> None:
    """Track edit_file failures per file and nudge on repeated failures."""
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    if tc.name != "edit_file":
        return

    path = str(tc.args.get("path", ""))
    if not path:
        return

    if not outcome.get("success"):
        st.edit_failures_per_file[path] = st.edit_failures_per_file.get(path, 0) + 1
        st.total_edit_failures += 1
        failures = st.edit_failures_per_file[path]

        if failures == 3:
            inject_nudge(agent, st, turn, (
                f"[system] You have failed to edit `{path}` {failures} times. "
                "Your old_str is not matching the file content. STOP guessing and:\n"
                "1. Call `read_file` on the exact line range you want to edit.\n"
                "2. Copy the EXACT text from the read output into old_str.\n"
                "3. Include 2-3 lines of surrounding context for uniqueness."
            ), min_gap=0)
        elif failures >= 5:
            inject_nudge(agent, st, turn, (
                f"[system] CRITICAL: {failures} failed edits to `{path}`. "
                "You are struggling with this file. Consider:\n"
                "1. Are you editing the RIGHT file? Re-read the problem statement.\n"
                "2. Try a completely different approach or a different file.\n"
                "3. Write a small reproduction script to verify you understand "
                "the bug before editing."
            ), min_gap=0)
    else:
        st.edit_failures_per_file[path] = 0


def inject_consecutive_identical_nudge(
    agent: Agent, st: LoopState, *, turn: int = 0,
) -> None:
    """Inject a warning when the model repeats the same tool call."""
    if st.consecutive_identical >= 5:
        hint_files = sorted(st.problem_files)[:3] if st.problem_files else []
        hint = (
            f"\nThe problem mentions these files: {', '.join(hint_files)}. "
            "If you haven't edited one yet, do so NOW."
        ) if hint_files else ""
        inject_nudge(agent, st, turn, (
            f"[system] CRITICAL: You have repeated the EXACT same call "
            f"{st.consecutive_identical + 1} times. ONE MORE and this task "
            "will be terminated. You MUST do something DIFFERENT right now:\n"
            "- Call `edit_file` with your best guess fix.\n"
            "- Or respond with text to finish."
            f"{hint}"
        ), min_gap=0, force=True)
    else:
        inject_nudge(agent, st, turn, (
            f"[system] WARNING: You have made the EXACT same tool call "
            f"{st.consecutive_identical + 1} times in a row. You are stuck "
            "in a loop. You MUST try something different:\n"
            "1. If you need to edit a file, call `edit_file` now.\n"
            "2. If you already fixed the bug, respond with plain text.\n"
            "3. Try a completely different file or approach.\n"
            "Do NOT repeat the same call again."
        ), min_gap=0, force=True)


def check_read_only_spiral(
    agent: Agent, st: LoopState, is_bench: bool, turn: int,
) -> TaskResult | None:
    """Detect read-only spiral and force-finish."""
    if (
        is_bench
        and turn >= 25
        and not st.files_edited
        and st.commands_run == 0
        and st.turns_without_progress >= 20
    ):
        msg = f"read-only spiral detected after {turn} turns — force finishing"
        if agent.display:
            agent.display.warn(msg)
        return agent._build_result(st, success=False, error=msg, turn=turn)

    if is_bench and turn >= 50 and not st.files_edited:
        msg = f"no edits after {turn} turns — force finishing"
        if agent.display:
            agent.display.warn(msg)
        return agent._build_result(st, success=False, error=msg, turn=turn)

    return None
