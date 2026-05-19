"""Turn-budget injection, re-anchoring, and problem text caching for bench/yolo modes.

Phase transitions are handled by phase_machine.py.  This module provides
informational helpers that run alongside the phase machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from squishy.agent_safety import inject_nudge
from squishy.agent_state import LoopState

if TYPE_CHECKING:
    from squishy.agent import Agent


def inject_turn_budget(agent: Agent, st: LoopState, turn: int) -> None:
    """Inject turn-budget awareness every 10 turns (bench/yolo)."""
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    if turn % 10 != 0 or turn == 0:
        return
    remaining = agent.config.max_turns - turn
    inject_nudge(agent, st, turn, (
        f"[system] Turn {turn}/{agent.config.max_turns}. "
        f"{remaining} turns remaining."
    ))


def maybe_post_edit_pytest_nudge(agent: Agent, st: LoopState, turn: int) -> None:
    """v6c: surface the exact pytest invocation right after the first
    successful edit in bench/yolo mode.

    The legacy generic finish_plan gate already includes the pytest
    command, but only fires *after* the model attempts finish_plan —
    by which time the model has often locked into a no-test loop
    (rdt-670 in v6b: 7 finish_plan calls, never invoked pytest).  This
    helper inverts the order: as soon as we know an edit succeeded, we
    tell the model what to run next, before it has the opportunity to
    drift.  Once-per-task; informational (no force-finish) so it
    composes cleanly with the existing v5/v6b/F5/generic gates.
    """
    if agent.config.permission_mode not in ("bench", "yolo"):
        return
    if st.post_edit_pytest_nudge_sent:
        return
    if st.last_edit_turn <= 0:
        return  # No successful edit yet.
    if not st.fail_to_pass_tests:
        return  # No actionable test command to suggest.

    # v6e: prefer the harness-supplied test_cmd (V2's install_config.test_cmd)
    # over hardcoded pytest.  For pytest-shaped runners, append F2P IDs as
    # space-separated args (the long-standing v6c behavior).  For foreign
    # runners (npm test, ./vendor/bin/phpunit, cargo test) we don't know the
    # syntax for selecting individual tests, so suggest the bare command and
    # tell the agent to verify the relevant F2P tests passed in its output.
    base_cmd = st.test_cmd or "pytest"
    if "pytest" in base_cmd:
        full_cmd = f"{base_cmd} " + " ".join(st.fail_to_pass_tests)
    else:
        full_cmd = base_cmd  # non-pytest runner — agent picks individual selection
    preview_count = min(5, len(st.fail_to_pass_tests))
    preview = ", ".join(f"`{t}`" for t in st.fail_to_pass_tests[:preview_count])
    more = (
        f" (and {len(st.fail_to_pass_tests) - preview_count} more)"
        if len(st.fail_to_pass_tests) > preview_count else ""
    )
    content = (
        "[system] You just landed your first edit. Before calling "
        "`finish_plan`, run the FAIL_TO_PASS tests to verify the fix:\n\n"
        f"```\n{full_cmd}\n```\n\n"
        f"Targets: {preview}{more}.  If they pass, *then* call "
        "`finish_plan(status=\"success\")`.  If they fail, read the "
        "error and edit again — do not call `finish_plan` until at "
        "least one F2P test passes."
    )
    # v6d: only set the one-shot flag when inject_nudge actually
    # appends. A min_gap rejection should let the next turn re-attempt
    # rather than burning the one-shot. min_gap lowered 3 → 1 because
    # this nudge is informational, once-per-task, and fires only after
    # a constructive event (successful edit) — back-to-back nudges are
    # acceptable when the prior nudge was an unrelated gate firing.
    if inject_nudge(agent, st, turn, content, min_gap=1):
        st.post_edit_pytest_nudge_sent = True


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
    inject_nudge(agent, st, turn, content, min_gap=3)


def cache_problem_text(agent: Agent, st: LoopState) -> None:
    """Cache problem text at loop start before compaction can destroy it."""
    # v6b: prefer the full F2P list from the bench harness (passed via
    # ``tool_ctx.notes['fail_to_pass_tests']``).  The prompt's visible
    # "## Failing Tests" section caps at 5 entries for readability,
    # which silently truncated st.fail_to_pass_tests when the harness
    # set 6+.  When notes are absent (interactive REPL), fall back to
    # the regex below.
    notes_f2p: list[str] = []
    notes = getattr(getattr(agent, "tool_ctx", None), "notes", None) or {}
    raw = notes.get("fail_to_pass_tests") if isinstance(notes, dict) else None
    if isinstance(raw, str):
        import json as _json
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                notes_f2p = [str(x) for x in parsed]
        except (ValueError, TypeError):
            pass
    elif isinstance(raw, list):
        notes_f2p = [str(x) for x in raw]
    if notes_f2p:
        st.fail_to_pass_tests = notes_f2p

    # v6e: pull harness-supplied test_cmd (V2's install_config.test_cmd) so
    # maybe_post_edit_pytest_nudge can suggest the right runner for non-pytest
    # projects.  Same notes shape as fail_to_pass_tests; tolerate both raw
    # str and JSON-encoded str (forward-compat with future plumbing).
    raw_cmd = notes.get("test_cmd") if isinstance(notes, dict) else None
    if isinstance(raw_cmd, str) and raw_cmd.strip():
        st.test_cmd = raw_cmd.strip()

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
            # Extract FAIL_TO_PASS test identifiers from the prompt as
            # a fallback when notes didn't supply them (e.g. interactive
            # REPL or a harness that doesn't set tool_ctx.notes).
            if not st.fail_to_pass_tests and "## Failing Tests" in content:
                import re
                tests_section = content.split("## Failing Tests", 1)[1]
                # Stop at the next section header.
                next_section = tests_section.find("\n## ")
                if next_section > 0:
                    tests_section = tests_section[:next_section]
                # Extract test IDs from backtick-quoted bullet items.
                st.fail_to_pass_tests = re.findall(r"^- `([^`]+)`", tests_section, re.MULTILINE)
            break
