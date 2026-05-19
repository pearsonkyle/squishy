"""Bridge squishy's ``prompt_fn`` callback to ACP ``session/request_permission``.

Squishy's tool dispatch already calls ``prompt_fn(tool, args)`` whenever a
mode-gated tool needs approval. The legacy CLI implements that callback as a
prompt_toolkit dialog; the ACP path replaces it with a JSON-RPC round-trip so
the editor can render a native permission dialog instead.
"""
from __future__ import annotations

import logging
from typing import Any

from acp.schema import (
    PermissionOption,
    ToolCallUpdate,
)

from squishy.tools.base import Tool

log = logging.getLogger("squishy.acp.permissions")


def make_prompt_fn(conn: Any, session_id: str) -> Any:
    """Return an async ``prompt_fn`` that defers to the ACP client.

    The returned callable matches squishy's ``PromptFn`` signature:
    ``async (tool: Tool, args: dict) -> bool | ("feedback", str)``. For
    ``plan_task`` we surface a "revise" option that maps back to the
    ``("feedback", "")`` tuple so the agent loop falls into its
    decline-with-feedback branch — the model will then revise the plan
    based on the editor-supplied note.
    """

    async def prompt_fn(tool: Tool, args: dict[str, object]) -> Any:
        title = _summarize_tool(tool, args)
        is_plan = tool.name == "plan_task"
        options = [
            PermissionOption(kind="allow_once", name="Allow", option_id="allow_once"),
            PermissionOption(
                kind="allow_always", name="Always allow", option_id="allow_always",
            ),
            PermissionOption(kind="reject_once", name="Reject", option_id="reject_once"),
        ]
        if is_plan:
            # Editors that surface free-text feedback can wire it into this
            # option; the bare button still maps to a polite "revise" signal.
            options.append(
                PermissionOption(
                    kind="reject_once", name="Revise…", option_id="revise",
                ),
            )
        else:
            options.append(
                PermissionOption(
                    kind="reject_always", name="Always reject", option_id="reject_always",
                ),
            )

        tool_call = ToolCallUpdate(
            tool_call_id=f"perm-{id(args):x}",
            title=title,
            kind=_acp_kind_for(tool.name),
            status="pending",
            raw_input=args,
        )

        try:
            resp = await conn.request_permission(
                options=options, session_id=session_id, tool_call=tool_call,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("session/request_permission failed: %s", exc)
            return False

        outcome = getattr(resp, "outcome", None)
        if outcome is None:
            return False
        # DeniedOutcome has no option_id; AllowedOutcome / SelectedPermissionOutcome do.
        option_id = getattr(outcome, "option_id", None)
        if option_id == "revise" and is_plan:
            return ("feedback", "")
        return option_id in ("allow_once", "allow_always")

    return prompt_fn


def _summarize_tool(tool: Tool, args: dict[str, object]) -> str:
    """Build a short human-readable title for the permission dialog.

    Mirrors the brief used in squishy's TUI so the editor shows the same
    summary the CLI would.
    """
    if tool.name == "run_command":
        cmd = str(args.get("command", "")).strip()
        return f"Run shell command: {cmd[:120]}" if cmd else "Run shell command"
    if tool.name in ("write_file", "edit_file"):
        path = str(args.get("path", "?"))
        return f"{tool.name.replace('_', ' ').title()}: {path}"
    if tool.name == "plan_task":
        return "Approve plan"
    return tool.name


def _acp_kind_for(tool_name: str) -> str:
    if tool_name == "run_command":
        return "execute"
    if tool_name in ("write_file", "edit_file", "undo_edit"):
        return "edit"
    if tool_name in ("read_file", "list_directory"):
        return "read"
    if tool_name in ("search_files", "glob_files", "recall"):
        return "search"
    return "other"
