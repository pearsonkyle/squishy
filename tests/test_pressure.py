"""Edit pressure: when it fires, when it stays quiet, and where it rides.

The unit-level companion to `test_shell_edit_pressure.py`, which drives the
same machinery through a whole agent run.
"""

from __future__ import annotations

from squishy.tools import dispatch, pressure
from squishy.tools.base import ToolContext, ToolResult


def _ctx(tmp_path, **kw) -> ToolContext:
    return ToolContext(
        working_dir=str(tmp_path), permission_mode="bench", use_sandbox=False, **kw
    )


def test_the_clock_is_silent_for_the_first_half(tmp_path) -> None:
    """Pressure applied early is noise a model learns to skip — and the first
    half of a run is when exploring is the correct thing to be doing."""
    assert pressure.budget_notice(used=1, budget=50) == ""
    assert pressure.budget_notice(used=24, budget=50) == ""
    assert "[budget]" in pressure.budget_notice(used=25, budget=50)


def test_the_clock_escalates_from_reminder_to_instruction() -> None:
    half = pressure.budget_notice(used=25, budget=50)
    late = pressure.budget_notice(used=40, budget=50)
    assert "Stop investigating" in half
    assert "next call" in late
    assert "scores zero" in late


def test_an_unbounded_run_is_never_pressured() -> None:
    """Interactive sessions have no turn budget; inventing one would be a lie."""
    assert pressure.budget_notice(used=99, budget=0) == ""


def test_probes_fire_after_a_run_of_commands_with_no_edit(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    for _ in range(pressure.PROBE_LIMIT):
        pressure.record_outcome(ctx, "run_command", ToolResult(True))
    assert "[probes]" in pressure.pressure_note(ctx)


def test_an_edit_resets_the_probe_counter(tmp_path) -> None:
    """Re-running a reproduction *after* an edit is the correct move, so the
    counter measures commands since the last edit — not commands in total."""
    ctx = _ctx(tmp_path)
    for _ in range(pressure.PROBE_LIMIT):
        pressure.record_outcome(ctx, "run_command", ToolResult(True))
    pressure.record_outcome(ctx, "edit_file", ToolResult(True, data={"path": "a.py"}))
    assert ctx.probe_commands == 0
    assert pressure.pressure_note(ctx) == ""


def test_a_scratch_write_does_not_count_as_the_edit(tmp_path) -> None:
    """A /tmp repro script is the right move and never reaches the diff, so
    treating it as "the fix landed" switches the pressure off exactly when it
    is needed."""
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 40, 50
    pressure.record_outcome(
        ctx, "write_file", ToolResult(True, data={"path": "/tmp/r.py", "scratch": True})
    )
    assert not ctx.source_edited
    assert "[budget]" in pressure.pressure_note(ctx)


def test_a_failed_edit_does_not_count_as_the_edit(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    pressure.record_outcome(ctx, "edit_file", ToolResult(False, error="old_str not found"))
    assert not ctx.source_edited


async def test_the_notice_rides_on_the_result_of_the_call_that_earned_it(
    tmp_path,
) -> None:
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 45, 50
    res = await dispatch("list_directory", {"path": "."}, ctx)
    assert res.success
    assert "[budget]" in res.data["pressure"]


async def test_a_failing_call_carries_the_notice_in_its_error(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 45, 50
    res = await dispatch("read_file", {"path": "nope.py"}, ctx)
    assert not res.success
    assert "[budget]" in res.error
    assert "nope.py" in res.error, "the original error must survive"


async def test_pressure_never_overwrites_an_existing_note(tmp_path) -> None:
    """A read-loop warning and a budget notice must not clobber each other."""
    (tmp_path / "a.py").write_text("x = 1\n")
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 45, 50
    for _ in range(2):  # the second read is served from cache, with its own note
        res = await dispatch("read_file", {"path": "a.py", "offset": 0}, ctx)
    assert res.success
    assert "[budget]" in res.data["pressure"]
    assert "already read this file" in res.data["note"]
