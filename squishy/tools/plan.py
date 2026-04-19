"""plan_task tool — structured planning with user approval and step tracking.

When the agent is in plan mode (or any mode), it can call ``plan_task`` to
present a structured plan to the user.  The plan contains:

- problem: what needs to be solved
- solution: high-level approach
- steps: ordered implementation steps
- files: which files to create / modify

After the plan is displayed the user is asked whether to proceed.  If they
accept, the steps are stored in the ToolContext so subsequent turns can mark
them complete.
"""

from __future__ import annotations

from typing import Any

from squishy.tools.base import Tool, ToolContext, ToolResult


async def _plan_task(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    plan = args.get("plan")
    problem = args.get("problem")
    solution = args.get("solution")
    steps = args.get("steps")
    files_to_modify = args.get("files_to_modify")
    files_to_create = args.get("files_to_create")

    if not isinstance(problem, str) or not problem.strip():
        return ToolResult(False, error="`problem` is required (string)")
    if not isinstance(solution, str) or not solution.strip():
        return ToolResult(False, error="`solution` is required (string)")
    if not isinstance(steps, list) or not steps:
        return ToolResult(False, error="`steps` is required (non-empty list of strings)")

    # Build a structured plan dict that will be stored for tracking
    plan_data: dict[str, Any] = {
        "plan": plan or "",
        "problem": problem,
        "solution": solution,
        "steps": [{"description": s, "status": "pending"} for s in steps],
        "files_to_create": files_to_create or [],
        "files_to_modify": files_to_modify or [],
    }

    # Store the plan on the context for tracking across turns
    ctx.active_plan = plan_data  # type: ignore[attr-defined]

    # Build a human-readable display for the tool result
    display_lines = [
        f"Plan: {plan}" if plan else "",
        f"Problem: {problem}",
        f"Solution: {solution}",
        "",
        "Steps:",
    ]
    for i, step in enumerate(steps, 1):
        display_lines.append(f"  {i}. {step}")

    if files_to_create:
        display_lines.append("")
        display_lines.append("Files to create:")
        for f in files_to_create:
            display_lines.append(f"  + {f}")

    if files_to_modify:
        display_lines.append("")
        display_lines.append("Files to modify:")
        for f in files_to_modify:
            display_lines.append(f"  ~ {f}")

    display_text = "\n".join(line for line in display_lines if line is not None)

    return ToolResult(
        True,
        data=plan_data,
        display=display_text,
    )


async def _update_plan(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Mark a step as done, skipped, or in-progress."""
    step_index = args.get("step_index")
    status = args.get("status", "done")

    if not isinstance(step_index, int):
        return ToolResult(False, error="`step_index` is required (integer, 1-based)")

    plan: dict[str, Any] | None = getattr(ctx, "active_plan", None)
    if plan is None:
        return ToolResult(False, error="no active plan — call plan_task first")

    steps = plan.get("steps", [])
    idx = step_index - 1
    if idx < 0 or idx >= len(steps):
        return ToolResult(False, error=f"step_index {step_index} out of range (1-{len(steps)})")

    if status not in ("done", "skipped", "in-progress"):
        return ToolResult(False, error="`status` must be 'done', 'skipped', or 'in-progress'")

    steps[idx]["status"] = status

    # Build a progress summary
    total = len(steps)
    done = sum(1 for s in steps if s["status"] == "done")
    in_prog = sum(1 for s in steps if s["status"] == "in-progress")
    pending = sum(1 for s in steps if s["status"] == "pending")

    progress_lines = []
    for i, s in enumerate(steps, 1):
        status_icons = {"done": "✓", "in-progress": "▶", "skipped": "—", "pending": "○"}
        mark = status_icons.get(s["status"], "?")
        progress_lines.append(f"  {mark} {i}. {s['description']}")

    summary = (
        f"Step {step_index} → {status}. "
        f"Progress: {done}/{total} done, {in_prog} in-progress, {pending} pending"
    )

    return ToolResult(
        True,
        data={
            "step_index": step_index,
            "status": status,
            "progress": {
                "done": done,
                "in_progress": in_prog,
                "pending": pending,
                "total": total,
            },
            "steps": steps,
        },
        display=summary + "\n" + "\n".join(progress_lines),
    )


plan_task = Tool(
    name="plan_task",
    description=(
        "Present a structured plan to the user before making changes. "
        "Use this when the task is complex and benefits from planning first. "
        "The plan includes: problem statement, solution approach, implementation "
        "steps, and files to create/modify. The user will be asked to approve."
    ),
    parameters={
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "Short title/summary of the plan",
            },
            "problem": {
                "type": "string",
                "description": "What problem needs to be solved",
            },
            "solution": {
                "type": "string",
                "description": "High-level approach to solving the problem",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of implementation steps",
            },
            "files_to_create": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files that need to be created",
            },
            "files_to_modify": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Existing files that need to be modified",
            },
        },
        "required": ["problem", "solution", "steps"],
    },
    run=_plan_task,
    mutates=False,
)

update_plan = Tool(
    name="update_plan",
    description=(
        "Update the status of a step in the active plan. Call this after "
        "completing each implementation step to track progress. "
        "Use step_index (1-based) and status ('done', 'in-progress', or 'skipped')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "step_index": {
                "type": "integer",
                "description": "1-based index of the step to update",
            },
            "status": {
                "type": "string",
                "enum": ["done", "in-progress", "skipped"],
                "description": "New status for the step",
            },
        },
        "required": ["step_index", "status"],
    },
    run=_update_plan,
    mutates=False,
)

PLAN_TOOLS: list[Tool] = [plan_task, update_plan]

__all__ = ["plan_task", "update_plan", "PLAN_TOOLS"]
