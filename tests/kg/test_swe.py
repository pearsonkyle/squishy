"""The write side: edit tools, local-model resolution, and the SWE arms.

Everything here runs with no API key and no network — agents are constructed,
never run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from graphagent.agentkit import edit
from graphagent.agentkit.llm import DEFAULT_MODEL, resolve_model
from graphagent.agentkit.metrics import ToolTrace
from graphagent.agentkit.swe import build_swe_agent
from graphagent.graph.builder import build_graph

# -- edit tools --------------------------------------------------------------


def test_edit_file_replaces_a_unique_occurrence(sample_repo: Path) -> None:
    target = "demo/utils.py"
    before = (sample_repo / target).read_text(encoding="utf-8")
    name = before.split("def ", 1)[1].split("(", 1)[0]
    result = edit.edit_file(sample_repo, target, f"def {name}(", f"def {name}_x(")
    assert result.startswith("edited")
    assert f"def {name}_x(" in (sample_repo / target).read_text(encoding="utf-8")


def test_edit_file_reports_how_many_times_it_matched(sample_repo: Path) -> None:
    """Zero and many are opposite fixes; the message has to tell them apart."""
    missing = edit.edit_file(sample_repo, "demo/utils.py", "nope_not_here", "x")
    assert "not found" in missing

    path = sample_repo / "dup.py"
    path.write_text("a = 1\na = 1\n", encoding="utf-8")
    assert "appears 2 times" in edit.edit_file(sample_repo, "dup.py", "a = 1", "b = 1")


def test_write_file_allows_the_scratch_dir_it_recommends() -> None:
    """The agent is told to put repro scripts in /tmp — refusing that path is
    the instruct-then-block bug this whole harness exists to avoid."""
    scratch = Path(tempfile.gettempdir()) / "graphagent_repro_probe.py"
    root = Path(tempfile.mkdtemp())
    try:
        assert edit.write_file(root, str(scratch), "print(1)\n").startswith("wrote")
        assert scratch.read_text(encoding="utf-8") == "print(1)\n"
    finally:
        scratch.unlink(missing_ok=True)


def test_write_file_still_rejects_absolute_paths_elsewhere() -> None:
    root = Path(tempfile.mkdtemp())
    assert edit.write_file(root, "/etc/passwd", "x").startswith("error:")
    assert edit.write_file(root, "../escape.py", "x").startswith("error:")


def test_run_command_reports_exit_code_and_output(sample_repo: Path) -> None:
    out = edit.run_command(sample_repo, "echo hello; exit 3")
    assert "exit=3" in out
    assert "hello" in out


# -- model resolution --------------------------------------------------------


def test_hosted_model_resolves_to_a_plain_name(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GRAPHAGENT_BASE_URL", "SQUISHY_BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_model() == DEFAULT_MODEL
    assert resolve_model("gpt-4o") == "gpt-4o"


def test_local_endpoint_refuses_to_guess_a_model_id() -> None:
    """LM Studio JIT-loads whatever id it is handed; a default would pull a
    stray multi-gigabyte model off disk."""
    with pytest.raises(ValueError, match="model id is required"):
        resolve_model(base_url="http://localhost:1234/v1")


def test_local_endpoint_builds_a_chat_completions_model() -> None:
    model = resolve_model("ornith-1.0-35b", base_url="http://localhost:1234/v1")
    assert getattr(model, "model", None) == "ornith-1.0-35b"


# -- the arms ----------------------------------------------------------------


def _tool_names(agent) -> set[str]:
    return {t.name for t in agent.tools}


def test_baseline_swe_arm_can_edit_but_has_no_graph_tools(sample_repo: Path) -> None:
    names = _tool_names(build_swe_agent(sample_repo))
    assert {"read_file", "edit_file", "write_file", "run_command", "grep"} <= names
    assert not names & {"explore", "impact_of", "repo_map"}


def test_graph_swe_arm_adds_the_graph_tools_and_nothing_less(
    sample_repo: Path,
) -> None:
    graph = build_graph(sample_repo)
    names = _tool_names(build_swe_agent(sample_repo, graph))
    assert {"explore", "impact_of", "repo_map"} <= names
    # Same write tools as the baseline: the toolset difference must be
    # exploration only, or the patch-rate comparison means nothing.
    assert {"read_file", "edit_file", "write_file", "run_command"} <= names


def test_both_arms_share_the_task_framing(sample_repo: Path) -> None:
    baseline = build_swe_agent(sample_repo).instructions
    graph = build_swe_agent(sample_repo, build_graph(sample_repo)).instructions
    assert "Reproduce the failure before you fix it." in baseline
    assert "Reproduce the failure before you fix it." in graph
    assert "DO NOT crawl files" in graph
    assert "DO NOT crawl files" not in baseline


# -- trace -------------------------------------------------------------------


def test_trace_records_arguments_and_failures() -> None:
    trace = ToolTrace()
    trace.record("run_command", "pytest -q")
    trace.record("edit_file", "a.py  old_str='def f('", ok=False)
    assert trace.total == 2
    assert trace.failures == 1
    assert trace.by_tool == {"run_command": 1, "edit_file": 1}
    assert trace.commands[0] == "1: run_command pytest -q"
    assert trace.commands[1].startswith("2!: edit_file")


# -- loop breaking -----------------------------------------------------------


def _call(agent, tool_name: str, **kwargs) -> str:
    """Invoke a wrapped SDK tool the way the runner would."""
    import asyncio
    import json

    from agents.tool_context import ToolContext

    tool = next(t for t in agent.tools if t.name == tool_name)
    payload = json.dumps(kwargs)
    ctx = ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id="test",
        tool_arguments=payload,
    )
    return asyncio.run(tool.on_invoke_tool(ctx, payload))


def test_an_identical_repeat_says_so_in_the_tool_result(sample_repo: Path) -> None:
    """qiskit-terra-5662: the graph arm ran one dead-end explore three times,
    then quit without editing. The answer belongs in the result, not in an
    injected user turn."""
    agent = build_swe_agent(sample_repo, build_graph(sample_repo))
    first = _call(agent, "explore", query="slugify")
    second = _call(agent, "explore", query="slugify")
    assert "[repeat]" not in first
    assert "[repeat] Identical call #2" in second
    assert second.startswith(first[:50])


def test_repeating_a_command_is_not_flagged(sample_repo: Path) -> None:
    """Re-running a reproduction after an edit is the correct move, and its
    output genuinely does change."""
    agent = build_swe_agent(sample_repo)
    _call(agent, "run_command", command="echo hi")
    assert "[repeat]" not in _call(agent, "run_command", command="echo hi")


def test_a_missed_query_names_the_nearest_symbols(sample_repo: Path) -> None:
    from graphagent.agentkit import tools as impl

    out = impl.explore(build_graph(sample_repo), sample_repo, "slugfy")
    assert "Closest names" in out
    assert "slugify" in out


# -- turn pressure -----------------------------------------------------------


def test_budget_notice_is_silent_early_and_imperative_late() -> None:
    """qiskit-terra-5662: sixteen repro scripts, fifty turns, no edit. Nothing
    in the SDK tells the model the clock is running."""
    from graphagent.agentkit.swe import _budget_notice

    assert _budget_notice(10, 50) == ""
    mid = _budget_notice(25, 50)
    assert "Turn 25 of 50" in mid
    assert "Stop investigating" in mid
    late = _budget_notice(40, 50)
    assert "Make the edit on your next call" in late
    assert _budget_notice(1, 0) == ""


def test_pressure_rides_on_tool_results_and_stops_at_the_first_edit(
    sample_repo: Path,
) -> None:
    turns = {"n": 40}
    agent = build_swe_agent(
        sample_repo, progress=lambda: (turns["n"], 50)
    )
    assert "[budget]" in _call(agent, "grep", pattern="slugify")

    # A /tmp scratch file is the right move and never reaches the diff, so it
    # must not switch the pressure off.
    scratch = Path(tempfile.gettempdir()) / "graphagent_pressure_probe.py"
    try:
        _call(agent, "write_file", relative=str(scratch), content="x = 1\n")
        assert "[budget]" in _call(agent, "grep", pattern="title")
    finally:
        scratch.unlink(missing_ok=True)

    _call(agent, "write_file", relative="demo/new.py", content="x = 1\n")
    assert "[budget]" not in _call(agent, "grep", pattern="normalize")


# -- reading in the graph arm ------------------------------------------------


def test_only_a_file_over_the_read_cap_gets_an_outline(sample_repo: Path) -> None:
    """Measured, after shipping the opposite: a turn costs 8.5-11k prompt
    tokens (the transcript is resent) and an outline saves at most 3.7k, so an
    outline that forces a follow-up call is a net loss. Above the read cap the
    file truncates anyway and the extra call was already unavoidable."""
    big = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(200))
    (sample_repo / "demo" / "big.py").write_text(big, encoding="utf-8")
    agent = build_swe_agent(sample_repo, build_graph(sample_repo))

    small = _call(agent, "read_file", relative="demo/services.py")
    assert "outline" not in small
    assert 'return f"Hello' in small  # verbatim, one call

    out = _call(agent, "read_file", relative="demo/big.py")
    assert "outline" in out
    assert "symbol_source" in out
    assert "return 199" not in out  # no bodies


def test_an_explicit_range_still_returns_verbatim_lines(sample_repo: Path) -> None:
    """The escape hatch has to stay open, or this is instruct-then-block."""
    big = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(200))
    (sample_repo / "demo" / "big.py").write_text(big, encoding="utf-8")
    agent = build_swe_agent(sample_repo, build_graph(sample_repo))
    out = _call(agent, "read_file", relative="demo/big.py", start_line=1,
                end_line=6)
    assert "outline" not in out
    assert "return 0" in out


def test_unindexed_files_fall_back_to_a_real_read(sample_repo: Path) -> None:
    agent = build_swe_agent(sample_repo, build_graph(sample_repo))
    assert "# demo" in _call(agent, "read_file", relative="README.md")


def test_the_baseline_arm_is_untouched(sample_repo: Path) -> None:
    """The outline is a graph behavior; changing the baseline's reads would
    change the thing being measured."""
    big = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(200))
    (sample_repo / "demo" / "big.py").write_text(big, encoding="utf-8")
    agent = build_swe_agent(sample_repo)
    out = _call(agent, "read_file", relative="demo/big.py")
    assert "outline" not in out
    assert "return 0" in out


def test_a_chain_of_commands_with_no_edit_gets_told_so(sample_repo: Path) -> None:
    """cfn-lint-3965: 37 of 48 calls were `python -c` probes, each a small
    variation of the last, and the run hit the turn cap with no edit."""
    agent = build_swe_agent(sample_repo)
    for i in range(7):
        assert "[probes]" not in _call(agent, "run_command", command=f"echo {i}")
    assert "[probes] 8 commands" in _call(agent, "run_command", command="echo 8")


def test_an_edit_resets_the_probe_count(sample_repo: Path) -> None:
    """Re-running a reproduction after an edit is the correct move."""
    agent = build_swe_agent(sample_repo)
    for i in range(9):
        _call(agent, "run_command", command=f"echo {i}")
    _call(agent, "edit_file", relative="demo/utils.py",
          old_str="def unused_helper", new_str="def unused_helper2")
    assert "[probes]" not in _call(agent, "run_command", command="echo after")


def test_scratch_writes_do_not_reset_the_probe_count(sample_repo: Path) -> None:
    """A /tmp repro script is not progress toward a patch."""
    agent = build_swe_agent(sample_repo)
    scratch = Path(tempfile.gettempdir()) / "graphagent_probe_probe.py"
    try:
        for i in range(8):
            _call(agent, "run_command", command=f"echo {i}")
        _call(agent, "write_file", relative=str(scratch), content="x = 1\n")
        assert "[probes]" in _call(agent, "run_command", command="echo again")
    finally:
        scratch.unlink(missing_ok=True)
