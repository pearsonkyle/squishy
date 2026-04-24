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

from squishy.plan_state import (
    STATUS_ICONS,
    STEP_STATUSES,
    PlanEvidence,
    PlanState,
    PlanStep,
    save_plan,
)
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
    if not all(isinstance(step, str) and step.strip() for step in steps):
        return ToolResult(False, error="`steps` must contain only non-empty strings")

    plan_state = PlanState.create(
        plan=str(plan or ""),
        problem=problem.strip(),
        solution=solution.strip(),
        steps=[step.strip() for step in steps],
        files_to_create=[str(item) for item in files_to_create or []],
        files_to_modify=[str(item) for item in files_to_modify or []],
    )

    ctx.plan = plan_state
    ctx.pending_plan_evidence.clear()
    save_plan(ctx.working_dir, plan_state)

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

    display_text = "\n".join(display_lines)

    return ToolResult(
        True,
        data=plan_state.to_dict(),
        display=display_text,
    )


async def _update_plan(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Mark a step as done, skipped, in-progress, or blocked.

    Optionally append new steps via ``add_steps: list[str]``. Appended steps
    always land at the end of the plan with status=``pending``.
    """
    step_index = args.get("step_index")
    status = args.get("status", "done")
    note = args.get("note", "")
    add_steps = args.get("add_steps")

    if not isinstance(step_index, int):
        return ToolResult(False, error="`step_index` is required (integer, 1-based)")
    if not isinstance(note, str):
        return ToolResult(False, error="`note` must be a string")
    if status not in STEP_STATUSES:
        allowed = ", ".join(repr(item) for item in STEP_STATUSES)
        return ToolResult(False, error=f"`status` must be one of {allowed}")
    if status == "blocked" and not note.strip():
        return ToolResult(False, error="`note` is required when status is 'blocked'")
    if add_steps is not None and (
        not isinstance(add_steps, list)
        or not all(isinstance(s, str) and s.strip() for s in add_steps)
    ):
        return ToolResult(False, error="`add_steps` must be a list of non-empty strings")

    plan = ctx.plan
    if plan is None:
        return ToolResult(False, error="no active plan — call plan_task first")

    idx = step_index - 1
    if idx < 0 or idx >= len(plan.steps):
        return ToolResult(False, error=f"step_index {step_index} out of range (1-{len(plan.steps)})")

    evidence = [
        PlanEvidence.from_dict(item)
        for item in ctx.pending_plan_evidence
        if isinstance(item, dict)
    ]
    step = plan.update_step(
        step_index=idx,
        status=status,
        note=note.strip(),
        evidence=evidence,
    )
    ctx.pending_plan_evidence.clear()

    if add_steps:
        offset = len(plan.steps)
        for i, desc in enumerate(add_steps):
            plan.steps.append(
                PlanStep(id=f"step-{offset + i + 1}", description=desc.strip())
            )
    save_plan(ctx.working_dir, plan)

    progress = plan.progress()
    total = progress["total"]
    done = progress["done"]
    in_prog = progress["in_progress"]
    pending = progress["pending"]
    blocked = progress["blocked"]

    progress_lines = []
    for i, current_step in enumerate(plan.steps, 1):
        mark = STATUS_ICONS.get(current_step.status, "?")
        progress_lines.append(f"  {mark} {i}. {current_step.description}")

    summary = (
        f"Step {step_index} → {status}. "
        f"Progress: {done}/{total} done, {in_prog} in-progress, {blocked} blocked, {pending} pending"
    )
    if step.note:
        summary += f" Note: {step.note}"

    return ToolResult(
        True,
        data={
            "step_index": step_index,
            "status": status,
            "progress": progress,
            "steps": [item.to_dict() for item in plan.steps],
            "note": step.note,
            "evidence_count": len(step.evidence),
            "plan": plan.to_dict(),
        },
        display=summary + "\n" + "\n".join(progress_lines),
    )


plan_task = Tool(
    name="plan_task",
    description=(
        "**CRITICAL: For any task requiring file changes, call this FIRST before implementing.**\n\n"
        "Present a structured plan to the user before making changes.\n"
        "- Call this within your first 2 turns for complex tasks\n"
        "- For simple tasks (e.g., reading one file), you may skip this\n"
        "The plan includes: problem statement, solution approach, implementation "
        "steps, and files to create/modify. The file lists are optional and may be partial or empty if some targets are still uncertain.\n"
        "Structure your response as:\n"
        '```json\n'
        '{\n'
        '  "problem": "What needs to be fixed or implemented",\n'
        '  "solution": "High-level approach to solve it",\n'
        '  "steps": ["Step 1 description", "Step 2 description"],\n'
        '  "files_to_modify": ["file1.py", "file2.py"],\n'
        '  "files_to_create": ["new_file.py"]\n'
        '}\n'
        '```\n'
        "The user will be asked to approve before you proceed with implementation."
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
)

update_plan = Tool(
    name="update_plan",
    description=(
        "Update the status of a step in the active plan. Call this after "
        "completing each implementation step to track progress. "
        "Use step_index (1-based), status, and an optional note. Any files edited "
        "or commands run since the last update are attached as evidence. "
        "Pass `add_steps=[...]` to append newly-discovered steps to the end of "
        "the plan when the work is larger than originally scoped."
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
                "enum": ["done", "in-progress", "skipped", "blocked"],
                "description": "New status for the step",
            },
            "note": {
                "type": "string",
                "description": "Optional note or rationale for this status update",
            },
            "add_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of new step descriptions to append to the plan. "
                    "Use when you discover the work is larger than you originally scoped."
                ),
            },
        },
        "required": ["step_index", "status"],
    },
    run=_update_plan,
)


async def _get_plan(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Return the current plan as a structured dict, or {'plan': None}."""
    plan = ctx.plan
    if plan is None:
        return ToolResult(True, data={"plan": None}, display="No active plan.")
    data = plan.to_dict()
    lines = [f"Plan {plan.id} ({'approved' if plan.approved else 'proposed'}):"]
    for i, step in enumerate(plan.steps, 1):
        mark = STATUS_ICONS.get(step.status, "?")
        lines.append(f"  {mark} {i}. {step.description}")
    return ToolResult(True, data={"plan": data}, display="\n".join(lines))


get_plan = Tool(
    name="get_plan",
    description=(
        "Return the active plan's current state (problem, solution, steps with "
        "status, files, progress). Use this when you have lost track of where "
        "you are in a long task. Returns {'plan': null} if no plan exists."
    ),
    parameters={"type": "object", "properties": {}},
    run=_get_plan,
)


PLAN_TOOLS: list[Tool] = [plan_task, update_plan, get_plan]

__all__ = ["plan_task", "update_plan", "get_plan", "PLAN_TOOLS"]
