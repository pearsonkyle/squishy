"""Tool dispatch wrapper, plan approval, and evidence recording for the agent loop."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from squishy.agent_state import LoopState, brief, is_test_path
from squishy.plan_state import clear_plan, save_plan
from squishy.tools import ToolResult, dispatch

if TYPE_CHECKING:
    from squishy.agent import Agent
    from squishy.client import ToolCall


async def run_tool(agent: Agent, turn: int, tc: ToolCall) -> dict[str, Any]:
    """Dispatch a single tool call and handle display + plan approval."""
    tc_brief = brief(tc)
    if agent.display:
        agent.display.turn_header(turn, agent.config.max_turns, tc.name, tc_brief)

    if tc.name == "run_command" and agent.display:
        agent.display.command_line(str(tc.args.get("command", "")))

    if tc.name == "edit_file" and agent.display:
        old_str = str(tc.args.get("old_str", ""))
        new_str = str(tc.args.get("new_str", ""))
        if old_str and new_str:
            agent.display.edit_diff(str(tc.args.get("path", "")), old_str, new_str)

    # Re-sync the permission mode immediately before dispatch so a mid-turn
    # shift-tab (e.g. yolo→plan to abort a destructive command) is enforced on
    # the remaining in-flight tool calls, not only from the next turn. The
    # ModeCycler mutates config.permission_mode between awaits during dispatch.
    agent.tool_ctx.permission_mode = agent.config.permission_mode

    t0 = time.monotonic()
    outcome = await dispatch(tc.name, tc.args, agent.tool_ctx, prompt_fn=agent.prompt_fn)
    dt_ms = (time.monotonic() - t0) * 1000

    plan_approved = False
    if outcome.success and tc.name == "plan_task":
        outcome, plan_approved = await handle_plan_approval(agent, tc, outcome)

    record_plan_evidence(agent, tc, outcome)

    if agent.display:
        if tc.name == "plan_task":
            pass  # Panel already rendered in handle_plan_approval.
        elif outcome.success and tc.name == "update_plan":
            plan = agent.tool_ctx.plan
            if plan:
                agent.display.plan_progress([step.to_dict() for step in plan.steps])
        else:
            agent.display.tool_result(
                outcome.success, outcome.display or outcome.error, dt_ms
            )

        if outcome.success and tc.name == "write_file":
            agent.display.write_preview(
                str(tc.args.get("path", "?")), str(tc.args.get("content", ""))
            )
        if tc.name == "run_command" and outcome.data.get("exit_code") is not None:
            agent.display.command_output(outcome.data)
        if outcome.success:
            if tc.name == "write_file":
                agent.display.stats.files_created.add(str(tc.args.get("path", "?")))
            elif tc.name == "edit_file":
                agent.display.stats.files_edited.add(str(tc.args.get("path", "?")))
            elif tc.name == "run_command":
                agent.display.stats.commands_run += 1

    append_tool_result(
        agent, tc,
        message=outcome.to_message(agent.tool_ctx.max_tool_output_chars),
    )

    # F1b: stamp the read path on the tool message so snip_old_tool_results
    # can build an accurate stub without scraping the (possibly truncated)
    # JSON body. Use the path string as-given by the agent — relative paths
    # are most useful to the model anyway.
    if tc.name == "read_file" and outcome.success and agent.messages:
        last_msg = agent.messages[-1]
        if last_msg.get("role") == "tool":
            last_msg["_squishy_read_path"] = str(tc.args.get("path", ""))

    # B3: invalidate prior recall results that mention this exact file
    # path once the agent has actually read the file.  Recall returns
    # path/symbol summaries for ~10 entries; once read_file lands on
    # one of those paths, the recall hit is redundant.
    if tc.name == "read_file" and outcome.success:
        _invalidate_superseded_recall(agent, str(tc.args.get("path", "")))

    # In bench mode, plan approval is NOT terminal — nudge the model to execute.
    if plan_approved and agent.config.permission_mode in ("bench", "yolo"):
        # Gate the EXECUTE nudge on prior exploration. If the agent hasn't
        # read at least one test file AND one non-test source file, push it
        # to explore first instead of jumping straight to edits — wrong-fix
        # patches frequently trace to skipping the test-file read.
        read_paths = list(agent.tool_ctx.files_read.keys())
        read_test = any(is_test_path(p) for p in read_paths)
        read_source = any(not is_test_path(p) for p in read_paths)
        if not (read_test and read_source):
            missing = []
            if not read_test:
                missing.append("a failing-test file")
            if not read_source:
                missing.append("a source file you plan to edit")
            agent.messages.append({"role": "user", "content": (
                f"[system] Plan approved. Read {' and '.join(missing)} first, "
                "then `edit_file` to implement the fix."
            )})
        else:
            agent.messages.append({"role": "user", "content": (
                "[system] Plan approved. Execute it: `edit_file` to implement "
                "the fix, `run_command` to verify, then `update_plan` for each "
                "step you actually finished."
            )})

    # Semantic anchoring: tag important tool results so they survive trimming.
    if agent.messages and agent.messages[-1].get("role") == "tool":
        should_anchor = (
            (tc.name == "run_command" and not outcome.success)
            or (tc.name == "edit_file" and outcome.success)
            or (tc.name == "search_files" and outcome.success and outcome.data.get("count", 0) > 0)
            or (tc.name == "read_file" and outcome.success
                and agent.tool_ctx.files_read_count.get(str(tc.args.get("path", "")), 0) <= 1)
        )
        if should_anchor:
            agent.messages[-1]["_squishy_anchor"] = True

    return {
        "success": outcome.success,
        "plan_approved": plan_approved,
        "data": outcome.data if isinstance(outcome.data, dict) else {},
        # Carried so the loop can report *why* a call failed, not just that
        # it did — a harness needs that to tell "model is confused" from
        # "tool is broken".
        "error": outcome.error or "",
    }


async def handle_plan_approval(
    agent: Agent, tc: ToolCall, outcome: ToolResult,
) -> tuple[ToolResult, bool]:
    """Handle plan_task approval flow. Returns (outcome, plan_approved)."""
    if agent.display:
        agent.display.plan_panel(outcome.data)
    # prompt_fn returns True/False, or a ("feedback", text) tuple to decline
    # with revision feedback (see cli.py prompt_fn).
    result: bool | str | tuple[str, str] = True  # auto-approve when non-interactive
    if agent.prompt_fn is not None:
        try:
            from squishy.tools.base import Tool
            result = await agent.prompt_fn(
                Tool(name="plan_task", description="", parameters={},
                     run=lambda *_: None),  # type: ignore[arg-type]
                tc.args,
            )
        except EOFError:
            result = False
        except KeyboardInterrupt:
            # User wants to abort the whole run. Clean up the pending plan
            # and re-raise so Agent.run() translates it into AgentCancelled.
            agent.tool_ctx.plan = None
            agent.tool_ctx.pending_plan_evidence.clear()
            agent.tool_ctx.plan_switch_prompted = False
            clear_plan(agent.tool_ctx.working_dir)
            raise
    if result is True:
        if agent.tool_ctx.plan is not None:
            agent.tool_ctx.plan.mark_approved()
            agent.tool_ctx.plan_switch_prompted = False
            save_plan(agent.tool_ctx.working_dir, agent.tool_ctx.plan)
        outcome = ToolResult(
            True,
            data={
                **outcome.data,
                "approved": True,
                "plan": agent.tool_ctx.plan.to_dict() if agent.tool_ctx.plan is not None else {},
            },
            display=outcome.display,
        )
        return outcome, True

    # Declined — include user feedback if provided. The interactive prompt
    # returns a ("feedback", text) tuple for "type feedback to revise"; unpack
    # it so the revision guidance actually reaches the model (was dropped).
    if isinstance(result, tuple) and len(result) == 2 and result[0] == "feedback":
        feedback = str(result[1])
    elif isinstance(result, str):
        feedback = result
    else:
        feedback = ""
    agent.tool_ctx.plan = None
    agent.tool_ctx.pending_plan_evidence.clear()
    agent.tool_ctx.plan_switch_prompted = False
    clear_plan(agent.tool_ctx.working_dir)
    if feedback:
        error_msg = (
            f"Plan declined by user. Their feedback:\n\n{feedback}\n\n"
            "Revise the plan to address this feedback, then call plan_task again."
        )
    else:
        error_msg = "Plan declined by user. Ask what they'd like changed, or propose a new approach."
    return ToolResult(False, error=error_msg), False


def record_plan_evidence(agent: Agent, tc: ToolCall, outcome: ToolResult) -> None:
    """Record tool outcome as plan evidence when an approved plan is active."""
    exit_code = outcome.data.get("exit_code")
    ran_command = tc.name == "run_command" and exit_code is not None
    if not (outcome.success or ran_command):
        return
    if agent.tool_ctx.plan is None or not agent.tool_ctx.plan.approved:
        return
    if tc.name in ("write_file", "edit_file"):
        agent.tool_ctx.pending_plan_evidence.append({
            "kind": tc.name,
            "path": str(tc.args.get("path", "")),
            "detail": "created or rewrote file" if tc.name == "write_file" else "edited existing file",
        })
    elif tc.name == "run_command":
        data = outcome.data
        agent.tool_ctx.pending_plan_evidence.append({
            "kind": "run_command",
            "command": str(tc.args.get("command", "")),
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "detail": str(data.get("stderr") or data.get("stdout") or "").strip()[:300],
        })


def append_tool_result(agent: Agent, tc: ToolCall, message: str) -> None:
    agent.messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "name": tc.name,
        "content": message,
    })


def _invalidate_superseded_recall(agent: Agent, read_path: str) -> None:
    """Replace prior recall tool results that explicitly mention *read_path*
    with a one-line stub.  Exact-path match only — substring fuzzing risks
    invalidating useful entries that share a parent directory name.
    """
    if not read_path:
        return
    # removeprefix, not lstrip: lstrip("./") strips ANY leading '.'/'/' chars,
    # so ".github/x.py" → "github/x.py" and "../x.py" → "x.py" (wrong file).
    norm = read_path.replace("\\", "/").removeprefix("./")
    for m in agent.messages[:-1]:
        if m.get("role") != "tool" or m.get("name") != "recall":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        if content.startswith("[recall result superseded"):
            continue
        # Look for `"path": "..."` JSON entries that exactly match.
        # The recall tool emits paths in its result objects.
        check_norm = content.replace("\\", "/")
        # Quoted-path match guards against partial substring collisions.
        if (
            f'"{norm}"' in check_norm
            or f'"./{norm}"' in check_norm
            or f': "{norm}"' in check_norm
        ):
            m["content"] = (
                f"[recall result superseded by read_file({read_path})]"
            )


def track_tool_outcome(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    turn: int = 0,
) -> None:
    """Update loop state based on a tool call outcome.

    ``turn`` (default 0 for callers that don't care) stamps
    ``last_edit_turn`` / ``last_f2p_test_turn`` for the v2 auto-pytest
    finish gate. Default-0 keeps the gate inert when it isn't passed,
    so existing callers stay safe.
    """
    if outcome["success"]:
        st.consecutive_errors = 0
        if tc.name in ("read_file", "list_directory", "search_files"):
            if tc.name == "read_file":
                rpath = str(tc.args.get("path", ""))
                st.recent_edit_fail_files.discard(rpath)
                abs_rpath = os.path.join(agent.tool_ctx.working_dir, rpath)
                try:
                    abs_rpath = os.path.realpath(abs_rpath)
                except OSError:
                    pass
                agent.tool_ctx.edit_fail_files.discard(abs_rpath)
        elif tc.name == "recall":
            agent.consecutive_reads_without_recall = 0
        if tc.name == "write_file":
            st.files_created.add(str(tc.args.get("path", "?")))
        elif tc.name == "edit_file":
            old = str(tc.args.get("old_str", ""))
            new = str(tc.args.get("new_str", ""))
            if old and old == new:
                st.total_quality_violations += 1
            else:
                st.files_edited.add(str(tc.args.get("path", "?")))
                st.last_edit_turn = turn
                # Clear edit-fail exemption after successful edit.
                edit_path = str(tc.args.get("path", ""))
                abs_edit = os.path.join(agent.tool_ctx.working_dir, edit_path)
                try:
                    abs_edit = os.path.realpath(abs_edit)
                except OSError:
                    pass
                agent.tool_ctx.edit_fail_files.discard(abs_edit)
                st.recent_edit_fail_files.discard(edit_path)
                # Cooldown: a real edit means the agent is making progress
                # again — decay accumulated quality violations so old
                # exploration-phase noise doesn't trip the post-edit
                # force-finish gate (was hitting at violation 4 even after
                # the agent recovered and edited).
                if st.total_quality_violations > 0:
                    st.total_quality_violations = max(
                        0, st.total_quality_violations - 2,
                    )
        elif tc.name == "run_command":
            st.commands_run += 1
            cmd = str(tc.args.get("command", ""))
            from squishy.agent_state import (
                distinct_f2p_files,
                f2p_files_in_command,
                is_test_command,
                test_covers_fail_to_pass,
            )
            exit_code = outcome.get("data", {}).get("exit_code")
            # v2: stamp the F2P-test turn whenever pytest hits any F2P file,
            # regardless of pass/fail — the auto-pytest finish gate just needs
            # to know the agent attempted verification since its last edit.
            if is_test_command(cmd) and f2p_files_in_command(
                cmd, st.fail_to_pass_tests,
            ):
                st.last_f2p_test_turn = turn
                # v5: capture failing F2P test IDs + assertion errors for
                # the pre-finish partial-pass gate.  Runs regardless of
                # exit_code because pytest exits non-zero on failure but
                # the structured ``test_summary`` (failed/errors counts)
                # is the source of truth.  Filter to F2P-listed tests so
                # the gate message stays signal-rich.
                test_summary = outcome.get("data", {}).get("test_summary")
                if test_summary and (
                    test_summary.get("failed", 0) > 0
                    or test_summary.get("errors", 0) > 0
                ):
                    f2p_set = set(st.fail_to_pass_tests)
                    captured: list[dict[str, str]] = []
                    for fail in test_summary.get("failures", []):
                        tid = str(fail.get("test", ""))
                        # Prefix-match in either direction: handles
                        # parametrized tests (test_foo[3d]) where F2P
                        # names just the base, and handles F2P entries
                        # that include params the test command omits.
                        is_f2p = tid in f2p_set or any(
                            tid.startswith(f) or f.startswith(tid)
                            for f in st.fail_to_pass_tests
                        )
                        if is_f2p:
                            captured.append({
                                "test": tid,
                                "error": str(fail.get("error", "")),
                            })
                    st.last_f2p_failures = captured[:5]
                    # v6b: when pytest reported errors but produced no
                    # parseable per-test failure lines, surface a flag
                    # so check_finish_plan_gate can emit a "fix
                    # collection first" hint instead of falling silent.
                    st.last_f2p_collection_error = (
                        not captured and test_summary.get("errors", 0) > 0
                    )
                elif test_summary and test_summary.get("failed", 0) == 0 \
                        and test_summary.get("errors", 0) == 0:
                    # Clean run — clear stale failures so the finish_plan
                    # gate doesn't re-cite stale IDs after an edit fixes
                    # the previously-failing tests.
                    st.last_f2p_failures = []
                    st.last_f2p_collection_error = False
            if (
                exit_code == 0
                and is_test_command(cmd)
                and test_covers_fail_to_pass(cmd, st.fail_to_pass_tests)
            ):
                # Guard against false test-pass: if structured parsing found
                # failures in the output, don't declare pass even if exit_code=0.
                test_summary = outcome.get("data", {}).get("test_summary")
                if test_summary and test_summary.get("failed", 0) > 0:
                    pass  # False positive — failures detected in output
                elif st.files_edited:
                    # F5: record which F2P file(s) this command covered.
                    # `test_passed_after_edit` only flips True when *every*
                    # distinct F2P file has been exercised by a passing run.
                    # In bench mode the phase machine performs the same
                    # accounting against PhaseState; the LoopState copy is
                    # kept here so non-bench callers (and diagnostics) see
                    # the same view.
                    covered = f2p_files_in_command(cmd, st.fail_to_pass_tests)
                    st.f2p_files_covered.update(covered)
                    needed = distinct_f2p_files(st.fail_to_pass_tests)
                    if not needed or st.f2p_files_covered >= needed:
                        st.test_passed_after_edit = True
                else:
                    # Tests pass BEFORE any edits — the workspace already has
                    # the old code, so the eval harness will apply different
                    # test expectations.  Warn the agent.
                    from squishy.agent_safety import inject_nudge
                    inject_nudge(agent, st, 0, (
                        "[system] The listed tests pass without any edit, so the "
                        "workspace lacks the updated test expectations. Do not "
                        "assume it's fixed — read the problem statement and make "
                        "the described source change."
                    ), min_gap=0)
    elif tc.name == "run_command":
        st.commands_run += 1
        st.consecutive_errors = 0
    else:
        st.consecutive_errors += 1
        if tc.name == "edit_file":
            edit_path = str(tc.args.get("path", ""))
            if edit_path:
                st.recent_edit_fail_files.add(edit_path)
                abs_edit = os.path.join(agent.tool_ctx.working_dir, edit_path)
                try:
                    abs_edit = os.path.realpath(abs_edit)
                except OSError:
                    pass
                agent.tool_ctx.edit_fail_files.add(abs_edit)
                if abs_edit in agent.tool_ctx.files_read_count:
                    agent.tool_ctx.files_read_count[abs_edit] = min(
                        agent.tool_ctx.files_read_count[abs_edit], 2
                    )
