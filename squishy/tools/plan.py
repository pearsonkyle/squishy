"""Plan-mode tools.

- ``exit_plan_mode`` — model submits a structured plan; user is shown a Rich
  panel and prompted to approve. On approval the permission mode flips to
  ``edits`` and the PlanTracker is populated so implementation steps can be
  marked off live.
- ``update_plan_step`` — model marks a step as in_progress / done / skipped
  during execution. The Display renders the updated status list.
"""

from __future__ import annotations

from typing import Any

from rich.panel import Panel

from squishy.tools.base import Tool, ToolContext, ToolResult


async def _exit_plan_mode(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan = str(args.get("plan", "")).strip()
    problem = str(args.get("problem", "")).strip()
    solution_steps = _as_str_list(args.get("solution_steps"))
    files_create = _as_str_list(args.get("files_create"))
    files_modify = _as_str_list(args.get("files_modify"))
    implementation_steps = _as_str_list(args.get("implementation_steps"))

    if not plan:
        return ToolResult(False, error="`plan` is required (one-paragraph summary)")
    if not implementation_steps:
        return ToolResult(
            False,
            error="`implementation_steps` must be a non-empty list of ordered steps",
        )

    display = ctx.display
    if display is not None:
        _render_plan_panel(
            display,
            plan=plan,
            problem=problem,
            solution_steps=solution_steps,
            files_create=files_create,
            files_modify=files_modify,
            implementation_steps=implementation_steps,
        )

    # Ask the user.
    approved = False
    if ctx.approve_fn is not None:
        try:
            approved = await ctx.approve_fn(
                "Approve this plan and switch to edits mode? [y/N] "
            )
        except Exception:  # noqa: BLE001
            approved = False
    else:
        # No approve_fn wired — typically non-interactive.
        return ToolResult(
            False,
            error="no interactive prompt available to approve the plan",
        )

    if not approved:
        if display is not None:
            display.info("plan declined — staying in plan mode. refine and re-submit.")
        return ToolResult(
            True,
            data={
                "approved": False,
                "note": "user declined; refine the plan and call exit_plan_mode again.",
            },
            display="plan declined",
        )

    # Approved: populate tracker and flip mode.
    if ctx.tracker is not None:
        ctx.tracker.populate(plan=plan, problem=problem, descriptions=implementation_steps)
    if ctx.cfg_ref is not None:
        ctx.cfg_ref.permission_mode = "edits"
    ctx.permission_mode = "edits"

    if display is not None:
        display.info("[bold green]plan approved → mode: edits[/]")
        display.plan_status()

    return ToolResult(
        True,
        data={
            "approved": True,
            "mode": "edits",
            "steps": len(implementation_steps),
        },
        display=f"approved · {len(implementation_steps)} steps",
    )


async def _update_plan_step(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    index = args.get("index")
    status = args.get("status")
    note = str(args.get("note", "")).strip()

    if not isinstance(index, int):
        return ToolResult(False, error="`index` must be an integer")
    if not isinstance(status, str):
        return ToolResult(False, error="`status` must be one of: in_progress, done, skipped")

    if ctx.tracker is None or not ctx.tracker.is_active():
        return ToolResult(False, error="no active plan; call exit_plan_mode first")

    ok, err = ctx.tracker.set_status(index, status)
    if not ok:
        return ToolResult(False, error=err)

    if ctx.display is not None:
        ctx.display.plan_status()
        if note:
            ctx.display.info(f"  note: {note}")

    return ToolResult(True, data={"index": index, "status": status}, display=f"step {index} → {status}")


def _as_str_list(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(x) for x in val if isinstance(x, (str, int, float))]


def _render_plan_panel(
    display: Any,
    *,
    plan: str,
    problem: str,
    solution_steps: list[str],
    files_create: list[str],
    files_modify: list[str],
    implementation_steps: list[str],
) -> None:
    console = display.console
    console.rule("[bold magenta]PLAN[/]", style="magenta")
    console.print(plan)

    if problem:
        console.rule("[bold red]PROBLEM[/]", style="red")
        console.print(problem)

    if solution_steps:
        console.rule("[bold green]SOLUTION[/]", style="green")
        for i, s in enumerate(solution_steps, 1):
            console.print(f"  {i}. {s}")

    if files_create or files_modify:
        console.rule("[bold blue]FILES[/]", style="blue")
        if files_create:
            console.print("[bold]create:[/]")
            for f in files_create:
                console.print(f"  + {f}")
        if files_modify:
            console.print("[bold]modify:[/]")
            for f in files_modify:
                console.print(f"  ~ {f}")

    console.rule("[bold yellow]IMPLEMENTATION STEPS[/]", style="yellow")
    for i, s in enumerate(implementation_steps):
        console.print(f"  {i}. {s}")
    console.print()


exit_plan_mode = Tool(
    name="exit_plan_mode",
    description=(
        "Submit the completed plan for the user's approval. Call this ONLY in plan mode, "
        "after reading enough of the codebase to be confident about the approach. "
        "On approval, the session switches to edits mode and the implementation_steps "
        "become a live checklist — mark progress with `update_plan_step`. "
        "If the user declines, refine and call again."
    ),
    parameters={
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "One-paragraph summary of the change and its intended outcome.",
            },
            "problem": {
                "type": "string",
                "description": "What specific problem this plan addresses.",
            },
            "solution_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "High-level ordered approach.",
            },
            "files_create": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths of new files to be created.",
            },
            "files_modify": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths of existing files to be modified.",
            },
            "implementation_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered, commit-sized steps that will be tracked live.",
            },
        },
        "required": ["plan", "implementation_steps"],
    },
    run=_exit_plan_mode,
    mutates=False,
)


update_plan_step = Tool(
    name="update_plan_step",
    description=(
        "Mark an implementation step as in_progress, done, or skipped. "
        "Only valid after exit_plan_mode has been approved."
    ),
    parameters={
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "0-based step index."},
            "status": {
                "type": "string",
                "enum": ["in_progress", "done", "skipped"],
            },
            "note": {
                "type": "string",
                "description": "Optional short note about the step.",
            },
        },
        "required": ["index", "status"],
    },
    run=_update_plan_step,
    mutates=False,
)


PLAN_TOOLS: list[Tool] = [exit_plan_mode, update_plan_step]
