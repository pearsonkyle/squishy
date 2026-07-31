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




def test_minimal_prompt_mentions_recall_only_with_an_index(tmp_path):
    project = detect_project(str(tmp_path))
    assert "recall" not in build_system_prompt(
        str(tmp_path), project, False, "bench", "minimal")




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






