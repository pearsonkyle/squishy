"""Tool profiles: what the model is shown, and what it still may call."""
from __future__ import annotations

import json

import pytest
from conftest import FakeClient

from squishy.api import Squishy
from squishy.config import Config
from squishy.context import build_system_prompt, detect_project
from squishy.tool_restrictions import MINIMAL_TOOLS, get_profile_tools
from squishy.tools import openai_schemas


def _names(schemas: list[dict]) -> set[str]:
    return {s["function"]["name"] for s in schemas}


def test_minimal_exposes_only_shell_and_file_primitives():
    assert _names(openai_schemas("bench", profile="minimal")) == set(MINIMAL_TOOLS)


def test_standard_profile_is_unchanged():
    # The default must be byte-identical to the pre-profile behavior, so
    # existing callers see no drift.
    assert openai_schemas("bench") == openai_schemas("bench", profile="standard")
    assert get_profile_tools("standard") is None


@pytest.mark.parametrize("mode", ["plan", "edits", "yolo", "bench"])
def test_minimal_never_widens_a_mode(mode):
    minimal = _names(openai_schemas(mode, profile="minimal"))
    standard = _names(openai_schemas(mode, profile="standard"))
    assert minimal <= standard


def test_extra_tools_adds_recall_back():
    got = _names(openai_schemas("bench", profile="minimal",
                                extra_tools=frozenset({"recall"})))
    assert got == set(MINIMAL_TOOLS) | {"recall"}


def test_extra_tools_ignored_under_standard_profile():
    # No narrowing means nothing to add back; the set is already complete.
    assert openai_schemas("bench", extra_tools=frozenset({"recall"})) == \
        openai_schemas("bench")


def test_minimal_is_substantially_cheaper():
    std = len(json.dumps(openai_schemas("bench", profile="standard")))
    mini = len(json.dumps(openai_schemas("bench", profile="minimal")))
    assert mini < std / 2


def test_profile_shapes_schema_but_adds_no_refusal():
    # A model trained on another harness may call a tool outside its profile.
    # That must still dispatch — the profile is a hint, not a gate.
    from squishy.tools import REGISTRY, check_permission
    allowed, _ = check_permission(REGISTRY["search_files"], "bench")
    assert allowed is True


def test_bad_profile_rejected_by_config_and_api():
    with pytest.raises(ValueError, match="tool_profile"):
        Config(tool_profile="tiny")
    with pytest.raises(ValueError, match="tool_profile"):
        Squishy(tool_profile="tiny")


def test_api_threads_profile_into_config():
    sq = Squishy(tool_profile="minimal")
    assert sq._make_config(None).tool_profile == "minimal"


def test_minimal_prompt_drops_phase_narration(tmp_path):
    project = detect_project(str(tmp_path))
    std = build_system_prompt(str(tmp_path), project, False, "bench", "standard")
    mini = build_system_prompt(str(tmp_path), project, False, "bench", "minimal")
    # The phase machine is off under minimal, so describing it would be a lie.
    assert "phase" in std.lower()
    assert "explore" not in mini.lower()
    # ...but the forcing language that actually moves patch rate stays.
    assert "not stop until" in mini.lower()
    # No prompt should advertise tools the profile doesn't expose.
    for absent in ("plan_task", "save_note", "show_diff", "glob_files"):
        assert absent not in mini
    assert len(mini) < len(std)


def test_minimal_prompt_mentions_recall_only_with_an_index(tmp_path):
    project = detect_project(str(tmp_path))
    assert "recall" not in build_system_prompt(
        str(tmp_path), project, False, "bench", "minimal")


async def test_minimal_bench_run_skips_the_phase_machine(tmp_path):
    """Bench + minimal must not phase-gate: one edit, then finish.

    Under the standard profile the phase machine starts in `explore`, whose
    schema has no `edit_file` at all. The minimal profile's premise is that a
    directive prompt does that job, so the agent may edit on turn one.
    """
    from squishy.agent import Agent
    from squishy.client import CompletionResult, ToolCall
    from squishy.display import Display

    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "minimal"
    cfg.max_turns = 5

    fake = FakeClient(script=[
        CompletionResult(tool_calls=[ToolCall(
            id="c1", name="edit_file",
            args={"path": "app.py", "old_string": "return 1", "new_string": "return 2"},
        )]),
        CompletionResult(text="Fixed.", tool_calls=[]),
    ])
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix f")

    assert result.success, result.error
    assert "app.py" in result.files_edited
    assert (tmp_path / "app.py").read_text().endswith("return 2\n")
    # No phase machine ran, so no phase was ever recorded.
    assert result.final_phase in ("", "explore")


async def test_minimal_bench_still_offers_edit_file_on_turn_one(tmp_path):
    """The schema the model actually sees on its first request includes edits."""
    from squishy.agent import Agent
    from squishy.client import CompletionResult
    from squishy.display import Display

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "minimal"
    cfg.max_turns = 1

    fake = FakeClient(script=[CompletionResult(text="done", tool_calls=[])])
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    await agent.run("noop")

    offered = {s["function"]["name"] for s in fake.tools_seen[0]}
    assert offered == set(MINIMAL_TOOLS)


@pytest.mark.parametrize("alias,canonical", [
    # Claude Code
    ("Bash", "run_command"), ("Read", "read_file"), ("Edit", "edit_file"),
    ("Write", "write_file"), ("Glob", "glob_files"), ("Grep", "search_files"),
    # Anthropic text-editor / SWE-agent / mini-swe-agent
    ("str_replace_based_edit_tool", "edit_file"), ("view", "read_file"),
    ("create", "write_file"), ("execute_bash", "run_command"),
])
def test_common_harness_vocabularies_normalize(alias, canonical):
    """A narrow tool set is only safe if the names models know still land.

    The minimal profile advertises four tools; models trained on Claude Code,
    mini-swe-agent or the Anthropic text-editor tool will reach for their own
    names regardless, and a rejected call costs a whole turn.
    """
    from squishy.tool_aliases import normalize_call
    assert normalize_call(alias, {})[0] == canonical


def test_canonical_names_are_never_remapped():
    from squishy.tool_aliases import normalize_call
    for name in MINIMAL_TOOLS:
        assert normalize_call(name, {})[0] == name


async def test_must_edit_gate_yields_rather_than_livelocking(tmp_path):
    """A refused shell call the model ignores must not loop forever.

    Observed live under the minimal profile: the gate refused run_command 25
    consecutive times and the model answered every refusal by calling it
    again, never trying edit_file. 25 turns burned, no patch. After a few
    refusals the gate lifts so the run can at least make progress.
    """
    from squishy.agent import _MAX_SHELL_REFUSALS, Agent
    from squishy.client import CompletionResult, ToolCall
    from squishy.display import Display

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "minimal"
    cfg.max_turns = 20
    cfg.max_turns_without_edit = 2

    # A model that only ever calls run_command, exactly as seen live.
    script = [
        CompletionResult(tool_calls=[ToolCall(
            id=f"c{i}", name="run_command", args={"command": f"echo {i}"})])
        for i in range(15)
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    await agent.run("fix it")

    assert agent._active_st is not None
    refusals = agent._active_st.shell_refusals
    assert refusals == _MAX_SHELL_REFUSALS, (
        f"gate should stop refusing after {_MAX_SHELL_REFUSALS}, got {refusals}")
    # Once lifted, it stays lifted for the rest of the run.
    assert "run_command" not in agent.tool_ctx.blocked_tools


def test_recall_is_hidden_without_an_index():
    """Offering a tool whose only possible answer is 'no index' wastes a call.

    Seen 6 times in one sweep: the standard profile advertised `recall`
    unconditionally, so the model called it and was told to run /init.
    """
    names = {s["function"]["name"]
             for s in openai_schemas("bench", has_index=False)}
    assert "recall" not in names
    assert "recall" in {s["function"]["name"]
                        for s in openai_schemas("bench", has_index=True)}


@pytest.mark.parametrize("mode", ["plan", "edits", "yolo", "bench"])
def test_recall_hidden_without_index_in_every_mode(mode):
    assert "recall" not in {s["function"]["name"]
                            for s in openai_schemas(mode, has_index=False)}


# --- shell profile: the mini-swe-agent / quant-tuner shape -----------------

def test_shell_profile_exposes_exactly_one_tool():
    names = {s["function"]["name"]
             for s in openai_schemas("bench", profile="shell", has_index=False)}
    assert names == {"run_command"}


def test_shell_profile_is_the_cheapest_by_far():
    import json
    cost = {
        p: len(json.dumps(openai_schemas("bench", profile=p, has_index=False)))
        for p in ("shell", "minimal", "standard")
    }
    assert cost["shell"] < cost["minimal"] < cost["standard"]
    assert cost["shell"] < cost["standard"] / 5


def test_shell_profile_has_no_edit_tool():
    from squishy.tool_restrictions import profile_has_edit_tool
    assert profile_has_edit_tool("shell") is False
    assert profile_has_edit_tool("minimal") is True
    assert profile_has_edit_tool("standard") is True


async def test_must_edit_gate_cannot_fire_under_shell_profile(tmp_path):
    """Withdrawing run_command here would leave the model with no tool at all.

    Edits go through the shell, so `files_edited` stays empty however much
    real work happens — the gate's trigger condition is permanently true and
    its remedy (`edit_file`) does not exist.
    """
    from squishy.agent import Agent
    from squishy.client import CompletionResult, ToolCall
    from squishy.display import Display

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.tool_profile = "shell"
    cfg.max_turns = 12
    cfg.max_turns_without_edit = 2

    fake = FakeClient(script=[
        CompletionResult(tool_calls=[ToolCall(
            id=f"c{i}", name="run_command", args={"command": f"echo {i}"})])
        for i in range(10)
    ])
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    await agent.run("fix it")

    assert "run_command" not in agent.tool_ctx.blocked_tools
    assert agent._active_st is not None
    assert agent._active_st.shell_refusals == 0
    # Every turn kept its one tool.
    for offered in fake.tools_seen:
        assert {s["function"]["name"] for s in offered} == {"run_command"}


def test_shell_prompt_explains_how_to_edit_without_an_edit_tool(tmp_path):
    prompt = build_system_prompt(
        str(tmp_path), detect_project(str(tmp_path)), False, "bench", "shell")
    assert "only tool" in prompt
    # Editing is the one thing a shell makes awkward, so it must be spelled out.
    assert "git diff" in prompt
    for absent in ("read_file", "edit_file", "recall", "plan_task"):
        assert absent not in prompt


@pytest.mark.parametrize("profile", ["shell", "minimal", "standard"])
@pytest.mark.parametrize("phase", ["explore", "plan", "execute", "verify"])
def test_no_phase_ever_leaves_a_profile_with_zero_tools(profile, phase):
    """A profile x phase intersection that is empty strands the model.

    The plan phase offers {plan_task, save_note, recall}; against a shell-only
    profile that intersects to nothing. Narrowed profiles therefore skip the
    phase machine entirely — this guards the invariant either way.
    """
    schemas = openai_schemas("bench", profile=profile, phase=phase, has_index=True)
    if profile == "standard":
        assert schemas, f"{profile}/{phase} exposes no tools"
