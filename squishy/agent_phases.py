"""Phase tracking, re-anchoring, and budget injection for bench/yolo modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from squishy.agent_safety import inject_nudge
from squishy.agent_state import LoopState, TaskResult, is_test_command

if TYPE_CHECKING:
    from squishy.agent import Agent
    from squishy.client import ToolCall


def update_phase(
    agent: Agent, st: LoopState,
    dispatched: list[tuple[ToolCall, dict[str, Any]]],
    *, turn: int = 0,
) -> TaskResult | None:
    """Update phase tracking after tool dispatch (bench/yolo only).

    Returns a TaskResult to force-finish, or None to continue.
    """
    if agent.config.permission_mode not in ("bench", "yolo"):
        return None

    had_edit = False
    had_test_command = False
    for tc, outcome in dispatched:
        if tc.name == "edit_file" and outcome.get("success"):
            had_edit = True
        elif tc.name == "run_command":
            cmd = str(tc.args.get("command", ""))
            if is_test_command(cmd):
                had_test_command = True

    # Phase transitions.
    if had_edit:
        st.phase = "fix"
        st.post_edit_read_turns = 0
    elif had_test_command and st.phase == "fix":
        st.phase = "verify"
        st.fix_verify_cycles += 1
    elif st.phase == "verify":
        st.phase = "fix"

    if st.phase == "explore" and not st.files_edited:
        st.explore_turns += 1

    # Post-edit read-only tracking.
    if st.files_edited and not had_edit and not had_test_command:
        st.post_edit_read_turns += 1
    elif had_edit or had_test_command:
        st.post_edit_read_turns = 0

    # Budget enforcement.
    if (
        st.phase == "explore"
        and st.explore_turns >= agent.config.max_explore_turns
        and not st.files_edited
    ):
        st.phase = "fix"
        inject_nudge(agent, st, turn, (
            f"[system] You have spent {st.explore_turns} turns exploring "
            "without making any edits. You MUST call `edit_file` NOW with "
            "your best fix attempt. A wrong fix that you iterate on is "
            "MUCH better than more exploration."
        ))

    if st.fix_verify_cycles >= agent.config.max_fix_verify_cycles and st.files_edited:
        if agent.display:
            agent.display.warn(
                f"phase: exhausted fix-verify budget ({st.fix_verify_cycles} cycles)"
            )
        # Only report success if the last test actually passed.
        exhausted_success = bool(st.test_passed_after_edit)
        return agent._build_result(
            st, success=exhausted_success,
            final_text="Fix applied. Agent exhausted edit-verify cycle budget.",
            turn=turn,
        )

    # Post-edit read-only warning.
    if st.post_edit_read_turns == agent.config.max_post_edit_read_turns:
        inject_nudge(agent, st, turn, (
            f"[system] WARNING: You have spent {st.post_edit_read_turns} "
            "turns only reading files after making edits. Either:\n"
            "1. Call `edit_file` with your next fix, OR\n"
            "2. Call `run_command` to verify your existing fix, OR\n"
            "3. Respond with text to finish.\n"
            "Do NOT read more files."
        ))

    return None


def inject_turn_budget(agent: Agent, st: LoopState, turn: int) -> None:
    """Inject turn-budget awareness every 10 turns (bench/yolo)."""
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    if turn % 10 != 0 or turn == 0:
        return
    remaining = agent.config.max_turns - turn
    guidance = {
        "explore": "You should be editing by now. Call edit_file with your best fix.",
        "fix": f"You have made {st.fix_verify_cycles} fix-verify cycles. If tests pass, STOP.",
        "verify": "Verify your fix and finish immediately if it passes.",
    }.get(st.phase, "")
    inject_nudge(agent, st, turn, (
        f"[system] Turn {turn}/{agent.config.max_turns}. "
        f"{remaining} turns remaining. Phase: {st.phase}. {guidance}"
    ))


def maybe_reanchor_problem(agent: Agent, st: LoopState, turn: int) -> None:
    """Periodically re-inject the problem statement to prevent drift."""
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    reanchor_interval = 8
    if turn < reanchor_interval or turn - st.last_reanchor_turn < reanchor_interval:
        return

    if not st.problem_text:
        return

    st.last_reanchor_turn = turn
    content = (
        f"[system] REMINDER — The original bug you are fixing:\n"
        f"{st.problem_text}\n\n"
        "Stay focused on THIS bug. Do not fix unrelated issues."
    )
    if st.fail_to_pass_tests:
        tests = ", ".join(f"`{t}`" for t in st.fail_to_pass_tests[:5])
        content += f"\n\nFailing tests to verify your fix: {tests}"
    agent.messages.append({"role": "user", "content": content})


def cache_problem_text(agent: Agent, st: LoopState) -> None:
    """Cache problem text at loop start before compaction can destroy it."""
    for msg in agent.messages:
        if msg.get("role") == "user" and not str(msg.get("content", "")).startswith("[system]"):
            content = str(msg.get("content", ""))
            if "## Problem" in content:
                # Cache the problem statement (up to 1500 chars to preserve
                # enough context for effective re-anchoring).
                text = content.split("## Problem", 1)[1].strip()
                if len(text) > 1500:
                    text = text[:1500] + "..."
                st.problem_text = text
            # Extract FAIL_TO_PASS test identifiers so we can verify the
            # agent runs the *correct* tests, not just any passing test.
            if "## Failing Tests" in content:
                import re
                tests_section = content.split("## Failing Tests", 1)[1]
                # Stop at the next section header.
                next_section = tests_section.find("\n## ")
                if next_section > 0:
                    tests_section = tests_section[:next_section]
                # Extract test IDs from backtick-quoted bullet items.
                st.fail_to_pass_tests = re.findall(r"^- `([^`]+)`", tests_section, re.MULTILINE)
            break
