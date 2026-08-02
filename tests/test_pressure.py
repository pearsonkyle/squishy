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
    assert "[probes]" in pressure.pressure_note(ctx)[0]


def test_an_edit_resets_the_probe_counter(tmp_path) -> None:
    """Re-running a reproduction *after* an edit is the correct move, so the
    counter measures commands since the last edit — not commands in total."""
    ctx = _ctx(tmp_path)
    for _ in range(pressure.PROBE_LIMIT):
        pressure.record_outcome(ctx, "run_command", ToolResult(True))
    pressure.record_outcome(ctx, "edit_file", ToolResult(True, data={"path": "a.py"}))
    assert ctx.probe_commands == 0
    assert pressure.pressure_note(ctx)[0] == ""


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
    assert "[budget]" in pressure.pressure_note(ctx)[0]


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
    assert "[budget]" in res.to_message()


async def test_the_notice_is_the_last_thing_the_model_reads(tmp_path) -> None:
    """Outside the JSON payload, not a key inside it.

    Buried in `data`, the notice was measurably ignored: 43-50 `[budget]` and
    74-85 `[probes]` notices across a 100-turn container run that still never
    edited a file.
    """
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 45, 50
    msg = (await dispatch("list_directory", {"path": "."}, ctx)).to_message()
    assert msg.rstrip().endswith("does not."), msg[-120:]
    assert '"pressure"' not in msg, "it must not be a JSON key any more"


async def test_a_big_payload_cannot_squeeze_the_notice_out(tmp_path) -> None:
    """`_short_json` snips the middle of an oversized payload; the notice has
    to be reserved out of the budget rather than compete for it."""
    (tmp_path / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(5000)))
    ctx = _ctx(tmp_path, max_tool_output_chars=1500)
    ctx.turns_used, ctx.turn_budget = 45, 50
    res = await dispatch("read_file", {"path": "big.py"}, ctx)
    msg = res.to_message(ctx.max_tool_output_chars)
    assert "[budget]" in msg


async def test_a_failing_call_carries_the_notice_too(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 45, 50
    res = await dispatch("read_file", {"path": "nope.py"}, ctx)
    assert not res.success
    msg = res.to_message()
    assert "[budget]" in msg
    assert "nope.py" in msg, "the original error must survive"


async def test_pressure_never_overwrites_an_existing_note(tmp_path) -> None:
    """A read-loop warning and a budget notice must not clobber each other."""
    (tmp_path / "a.py").write_text("x = 1\n")
    ctx = _ctx(tmp_path)
    ctx.turns_used, ctx.turn_budget = 45, 50
    for _ in range(2):  # the second read is served from cache, with its own note
        res = await dispatch("read_file", {"path": "a.py", "offset": 0}, ctx)
    assert res.success
    msg = res.to_message()
    assert "[budget]" in msg
    assert "already read this file" in msg


async def test_the_notice_is_reported_on_the_tool_event(tmp_path) -> None:
    """A brake you cannot see in the result file cannot be evaluated.

    cfn-lint-3965 ran 100 turns and produced no edit. "The model was warned
    fifty times and ignored it" and "the warning never fired" are opposite
    diagnoses and looked identical from the outside, because nothing carried
    the tags out of the loop.
    """
    from conftest import FakeClient

    from squishy.agent import Agent
    from squishy.client import CompletionResult, ToolCall
    from squishy.config import Config

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "shell"
    cfg.max_turns = 12
    cfg.use_sandbox = False
    events: list[dict] = []
    script = [
        CompletionResult(tool_calls=[ToolCall(id=f"c{i}", name="run_command",
                                              args={"command": f"ls {i}"})])
        for i in range(12)
    ]
    agent = Agent(cfg, FakeClient(script=script), display=None,  # type: ignore[arg-type]
                  on_event=events.append)
    await agent.run("fix it")

    tags = {t for e in events if e.get("type") == "tool" for t in e.get("pressure", [])}
    assert "probes" in tags
    assert "budget" in tags


async def test_an_identical_failing_edit_is_called_out(tmp_path) -> None:
    """qiskit-terra-5662 issued the same non-matching edit_file twice in a
    row and nothing told it the second could not work.

    Advice, never a refusal: `read_file`'s span cache does refuse repeats but
    deliberately forgets after trimming, so a long run loses the signal
    exactly when it needs it. A counter that never forgets is only safe
    because it does not block.
    """
    (tmp_path / "a.py").write_text("x = 1\n")
    ctx = _ctx(tmp_path)
    args = {"path": "a.py", "old_str": "nope\nnope\nnope\n", "new_str": "y"}

    first = await dispatch("edit_file", dict(args), ctx)
    assert not first.success
    assert "[repeat]" not in first.to_message()

    second = await dispatch("edit_file", dict(args), ctx)
    assert not second.success
    msg = second.to_message()
    assert "[repeat]" in msg
    assert "cannot succeed" in msg


async def test_a_repeated_successful_call_is_called_out_more_gently(
    tmp_path,
) -> None:
    build = tmp_path / "m.py"
    build.write_text("def f():\n    return 1\n")
    from squishy.graph import build_repo_graph
    build_repo_graph(tmp_path)
    ctx = _ctx(tmp_path)

    await dispatch("explore", {"query": "f"}, ctx)
    msg = (await dispatch("explore", {"query": "f"}, ctx)).to_message()
    assert "[repeat]" in msg
    assert "will not change" in msg


async def test_different_arguments_are_not_a_repeat(tmp_path) -> None:
    (tmp_path / "a.py").write_text("\n".join(f"x{i} = {i}" for i in range(50)))
    ctx = _ctx(tmp_path)
    await dispatch("read_file", {"path": "a.py", "offset": 0, "limit": 10}, ctx)
    res = await dispatch("read_file", {"path": "a.py", "offset": 10, "limit": 10}, ctx)
    assert "[repeat]" not in res.to_message(), "paging is not looping"


async def test_run_command_is_exempt(tmp_path) -> None:
    """Re-running a test after an edit is correct; shell.py has its own,
    output-aware counter for genuinely echoing commands."""
    ctx = _ctx(tmp_path)
    for _ in range(3):
        res = await dispatch("run_command", {"command": "true"}, ctx)
    assert "[repeat]" not in res.to_message()
