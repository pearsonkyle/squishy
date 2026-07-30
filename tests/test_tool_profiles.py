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
