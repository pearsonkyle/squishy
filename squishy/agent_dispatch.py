"""Tool dispatch wrapper, plan approval, and evidence recording for the agent loop."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from squishy.agent_state import LoopState, brief
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

    append_tool_result(agent, tc, message=outcome.to_message())

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
    }


async def handle_plan_approval(
    agent: Agent, tc: ToolCall, outcome: ToolResult,
) -> tuple[ToolResult, bool]:
    """Handle plan_task approval flow. Returns (outcome, plan_approved)."""
    if agent.display:
        agent.display.plan_panel(outcome.data)
    result: bool | str = True  # auto-approve when non-interactive
    if agent.prompt_fn is not None:
        try:
            from squishy.tools.base import Tool
            result = await agent.prompt_fn(
                Tool(name="plan_task", description="", parameters={},
                     run=lambda *_: None),  # type: ignore[arg-type]
                tc.args,
            )
        except (EOFError, KeyboardInterrupt):
            result = False
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

    # Declined — include user feedback if provided.
    feedback = result if isinstance(result, str) else ""
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


def track_tool_outcome(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
) -> None:
    """Update loop state based on a tool call outcome."""
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
                # Clear edit-fail exemption after successful edit.
                edit_path = str(tc.args.get("path", ""))
                abs_edit = os.path.join(agent.tool_ctx.working_dir, edit_path)
                try:
                    abs_edit = os.path.realpath(abs_edit)
                except OSError:
                    pass
                agent.tool_ctx.edit_fail_files.discard(abs_edit)
                st.recent_edit_fail_files.discard(edit_path)
        elif tc.name == "run_command":
            st.commands_run += 1
            cmd = str(tc.args.get("command", ""))
            from squishy.agent_state import is_test_command, test_covers_fail_to_pass
            if (
                st.files_edited
                and outcome.get("data", {}).get("exit_code") == 0
                and is_test_command(cmd)
                and test_covers_fail_to_pass(cmd, st.fail_to_pass_tests)
            ):
                # Guard against false test-pass: if structured parsing found
                # failures in the output, don't declare pass even if exit_code=0.
                test_summary = outcome.get("data", {}).get("test_summary")
                if test_summary and test_summary.get("failed", 0) > 0:
                    pass  # False positive — failures detected in output
                else:
                    st.test_passed_after_edit = True
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
