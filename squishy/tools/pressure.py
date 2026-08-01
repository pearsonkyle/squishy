"""Edit pressure, delivered in the tool result.

Two failure modes, both measured on SWE-rebench, both invisible to the model
without help:

* **The clock.** Nothing tells a model its turn budget is running out, so it
  spends a bounded budget the way it would spend an unbounded one.
  qiskit-terra-5662 wrote sixteen reproduction scripts and hit the 50-turn cap
  having never touched a source file — a run scored zero for want of one
  sentence.
* **Investigation that never converges.** cfn-lint-3965 spent 37 of its 48
  calls on `python -c` probes into one schema manager, each a small variation
  of the last. No two were identical, so the repeat detectors could not see
  it; only the run of them with nothing edited gives it away.

Both notices ride on the tool result rather than an injected `[system]` user
turn. That is the load-bearing rule of this loop (see `agent_safety`): a tool
result is causally paired with the call that produced it, is what the model is
trained to read, and cannot desynchronize the transcript.

`run_command` is deliberately not counted as a repeat by the probe brake's
sibling in `shell.py`, and re-running a reproduction *after* an edit is the
correct move — which is why the probe counter resets on every source edit
rather than counting commands for the whole run.
"""

from __future__ import annotations

from squishy.tools.base import ToolContext, ToolResult

# Consecutive commands with nothing edited before the harness says so. Eight
# is roughly where cfn-lint-3965's probe chain stopped producing new
# information and started restating the same query.
PROBE_LIMIT = 8

_EDIT_TOOLS = frozenset({"edit_file", "write_file", "undo_edit"})

_PROBES = (
    "[probes] {n} commands run and no source file changed yet. Experiments "
    "are not converging on their own — make the edit your evidence already "
    "supports, then re-run this command to check it."
)

_BUDGET_HALF = (
    "[budget] Turn {used} of {budget}, {left} left, and no source file has "
    "been edited yet. Stop investigating and change the code you already have "
    "evidence for — you can keep testing afterwards."
)

_BUDGET_LATE = (
    "[budget] Turn {used} of {budget}, {left} left, and no source file has "
    "been edited. Make the edit on your next call, using the best explanation "
    "you have. An unedited run scores zero; a partial fix does not."
)


def record_outcome(ctx: ToolContext, name: str, result: ToolResult) -> None:
    """Update the edit/probe counters from one completed tool call.

    Only a repo file counts as an edit. A reproduction script under /tmp is
    the right move and deliberately never reaches the diff, so treating it as
    "the edit landed" would switch the pressure off at exactly the moment it
    is needed.
    """
    if name in _EDIT_TOOLS:
        if result.success and not result.data.get("scratch"):
            ctx.source_edited = True
            ctx.probe_commands = 0
        return
    if name == "run_command":
        ctx.probe_commands += 1


def budget_notice(used: int, budget: int) -> str:
    """What to say to an agent running out of turns without having edited.

    Silent below half the budget: pressure applied too early is just noise
    the model learns to skip, and the first half of a run is when exploring
    is the correct thing to be doing. Half is a reminder; four fifths is an
    instruction.
    """
    if budget <= 0 or used <= 0 or used < budget // 2:
        return ""
    left = max(0, budget - used)
    template = _BUDGET_LATE if used >= budget * 4 // 5 else _BUDGET_HALF
    return template.format(used=used, budget=budget, left=left)


def pressure_note(ctx: ToolContext) -> str:
    """The notices this call has earned, or "" for the common case."""
    if ctx.source_edited:
        return ""
    parts: list[str] = []
    if ctx.probe_commands >= PROBE_LIMIT:
        parts.append(_PROBES.format(n=ctx.probe_commands))
    budget = budget_notice(ctx.turns_used, ctx.turn_budget)
    if budget:
        parts.append(budget)
    return "\n".join(parts)


def apply(ctx: ToolContext, name: str, result: ToolResult) -> ToolResult:
    """Record the call's outcome and attach any pressure notice to it."""
    record_outcome(ctx, name, result)
    note = pressure_note(ctx)
    if not note:
        return result
    if result.success:
        # Under its own key: appending to an existing `note`/`warning` would
        # let a read-loop warning and a budget notice overwrite each other.
        result.data["pressure"] = note
    else:
        result.error = f"{result.error}\n\n{note}" if result.error else note
    return result


__all__ = ["PROBE_LIMIT", "apply", "budget_notice", "pressure_note", "record_outcome"]
