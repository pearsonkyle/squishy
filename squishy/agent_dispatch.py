"""Tool dispatch wrapper and outcome tracking for the agent loop."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from squishy.agent_state import LoopState, brief
from squishy.tools import dispatch

if TYPE_CHECKING:
    from squishy.agent import Agent
    from squishy.client import ToolCall


async def run_tool(agent: Agent, turn: int, tc: ToolCall) -> dict[str, Any]:
    """Dispatch a single tool call and render it to the display."""
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
    # shift-tab (e.g. yolo→edits to abort a destructive command) is enforced on
    # the remaining in-flight tool calls, not only from the next turn. The
    # ModeCycler mutates config.permission_mode between awaits during dispatch.
    agent.tool_ctx.permission_mode = agent.config.permission_mode

    t0 = time.monotonic()
    outcome = await dispatch(tc.name, tc.args, agent.tool_ctx, prompt_fn=agent.prompt_fn)
    dt_ms = (time.monotonic() - t0) * 1000

    if agent.display:
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

    # Stamp the read path on the tool message so snip_old_tool_results can build
    # an accurate stub without scraping the (possibly truncated) JSON body. Use
    # the path string as-given by the agent — relative paths are most useful to
    # the model anyway.
    if tc.name == "read_file" and outcome.success and agent.messages:
        last_msg = agent.messages[-1]
        if last_msg.get("role") == "tool":
            last_msg["_squishy_read_path"] = str(tc.args.get("path", ""))

    # Invalidate prior recall results that mention this exact file path once the
    # agent has actually read the file. Recall returns path/symbol summaries for
    # ~10 entries; once read_file lands on one of those paths, the hit is
    # redundant.
    if tc.name == "read_file" and outcome.success:
        _invalidate_superseded_recall(agent, str(tc.args.get("path", "")))

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
        "data": outcome.data if isinstance(outcome.data, dict) else {},
        # Carried so the loop can report *why* a call failed, not just that it
        # did — a harness needs that to tell "model is confused" from "tool is
        # broken".
        "error": outcome.error or "",
    }


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
        check_norm = content.replace("\\", "/")
        # Quoted-path match guards against partial substring collisions.
        if (
            f'"{norm}"' in check_norm
            or f'"./{norm}"' in check_norm
            or f': "{norm}"' in check_norm
        ):
            m["content"] = f"[recall result superseded by read_file({read_path})]"


def track_tool_outcome(
    agent: Agent, st: LoopState, tc: ToolCall, outcome: dict[str, Any],
    turn: int = 0,
) -> None:
    """Update loop state based on a tool call outcome.

    Bookkeeping only — this used to also drive a phase machine and a
    FAIL_TO_PASS finish gate, both of which are gone.
    """
    if outcome["success"]:
        st.consecutive_errors = 0
        if tc.name == "read_file":
            rpath = str(tc.args.get("path", ""))
            st.recent_edit_fail_files.discard(rpath)
            agent.tool_ctx.edit_fail_files.discard(_abs(agent, rpath))
        if tc.name == "write_file":
            st.files_created.add(str(tc.args.get("path", "?")))
        elif tc.name == "edit_file":
            old = str(tc.args.get("old_str", ""))
            new = str(tc.args.get("new_str", ""))
            # A no-op edit is not progress; don't let it satisfy the
            # "have you changed anything" check.
            if old != new:
                edit_path = str(tc.args.get("path", "?"))
                st.files_edited.add(edit_path)
                st.last_edit_turn = turn
                agent.tool_ctx.edit_fail_files.discard(_abs(agent, edit_path))
                st.recent_edit_fail_files.discard(edit_path)
        elif tc.name == "run_command":
            st.commands_run += 1
    elif tc.name == "run_command":
        # A non-zero exit is information, not a harness error — a failing test
        # run is the normal case mid-task.
        st.commands_run += 1
        st.consecutive_errors = 0
    else:
        st.consecutive_errors += 1
        if tc.name == "edit_file":
            edit_path = str(tc.args.get("path", ""))
            if edit_path:
                st.recent_edit_fail_files.add(edit_path)
                abs_edit = _abs(agent, edit_path)
                agent.tool_ctx.edit_fail_files.add(abs_edit)
                # Let the model re-read a file it just failed to edit: the
                # read guard would otherwise refuse the exact call the edit
                # error tells it to make.
                if abs_edit in agent.tool_ctx.files_read_count:
                    agent.tool_ctx.files_read_count[abs_edit] = min(
                        agent.tool_ctx.files_read_count[abs_edit], 2
                    )


def _abs(agent: Agent, path: str) -> str:
    p = os.path.join(agent.tool_ctx.working_dir, path)
    try:
        return os.path.realpath(p)
    except OSError:
        return p
