"""The harness must never name a tool the model cannot see.

Seven bugs in this repo have had the same shape: the harness instructs an
action and then makes it impossible. Withdrawing `run_command` while telling
the model to edit. Refusing re-reads of content trimming had deleted. "Run the
failing tests you were given" when it was given none. `write_file` saying "put
it under /tmp" while the path guard refused /tmp. The bench prompt banning
reproduction scripts while the task prompt asked for one.

The cheapest recurring variant is prose that names a tool a narrow profile
does not expose. The model cannot follow the instruction and has no way to
find out why — it just burns turns. These tests close that variant for good.
"""

from __future__ import annotations

import re

from squishy.context import build_system_prompt, detect_project
from squishy.graph import build_repo_graph
from squishy.tool_restrictions import TOOL_PROFILES, profile_shows
from squishy.tools import REGISTRY, dispatch, openai_schemas
from squishy.tools.base import ToolContext

PROFILES = sorted(TOOL_PROFILES)

# Names that look like tools in prose. Anything matching `foo(` or `` `foo` ``
# that is also a real tool name has to be checked against the profile.
_MENTION = re.compile(r"`?([a-z_]+)`?\s*\(|`([a-z_]+)`")


def _mentioned_tools(text: str) -> set[str]:
    found: set[str] = set()
    for a, b in _MENTION.findall(text):
        name = a or b
        if name in REGISTRY:
            found.add(name)
    return found


def test_the_system_prompt_only_names_tools_the_profile_shows(tmp_path) -> None:
    """Checked for every profile, in every mode, with and without a graph."""
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    for has_graph in (False, True):
        if has_graph:
            build_repo_graph(tmp_path)
        for profile in PROFILES:
            for mode in ("edits", "yolo", "bench"):
                prompt = build_system_prompt(
                    str(tmp_path), detect_project(str(tmp_path)), False, mode, profile
                )
                for name in _mentioned_tools(prompt):
                    assert profile_shows(profile, name), (
                        f"{profile}/{mode} (graph={has_graph}) prompt names "
                        f"{name!r}, which that profile never shows"
                    )


def test_the_schema_and_the_prompt_agree(tmp_path) -> None:
    """A stronger form: measured against the schema actually sent.

    Prompt and schema are both derived from the same on-disk state, because
    that is the only way the comparison means anything — the pair the model
    receives has to agree, not two independently-configured versions of it.
    """
    from squishy.index import build_index, save_index
    from squishy.index.store import has_index

    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    build_repo_graph(tmp_path)
    save_index(tmp_path, build_index(str(tmp_path)))
    assert has_index(tmp_path)

    for profile in PROFILES:
        offered = {
            s["function"]["name"]
            for s in openai_schemas(
                "bench", profile=profile,
                extra_tools=frozenset({"recall"}),
                has_index=True, has_graph=True,
            )
        }
        prompt = build_system_prompt(
            str(tmp_path), detect_project(str(tmp_path)), False, "bench", profile
        )
        assert _mentioned_tools(prompt) <= offered, (
            f"{profile}: prompt names {_mentioned_tools(prompt) - offered} "
            "which is not in the schema it is sent with"
        )


async def test_a_long_read_does_not_offer_explore_to_a_profile_without_it(
    tmp_path,
) -> None:
    """`minimal` has a graph on disk but no `explore` in its schema."""
    body = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(400))
    (tmp_path / "huge.py").write_text(body, encoding="utf-8")
    build_repo_graph(tmp_path)

    for profile in ("minimal", "graph"):
        ctx = ToolContext(
            working_dir=str(tmp_path), permission_mode="bench", use_sandbox=False,
            tool_profile=profile, max_tool_output_chars=2000,
        )
        res = await dispatch("read_file", {"path": "huge.py"}, ctx)
        assert "outline" in res.data, profile
        mentions_explore = "explore()" in res.data["note"]
        assert mentions_explore == profile_shows(profile, "explore"), profile


async def test_an_explore_miss_does_not_offer_an_invisible_search_tool(
    tmp_path,
) -> None:
    """`graph` exposes `explore` but not `search_files`."""
    (tmp_path / "m.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    build_repo_graph(tmp_path)

    for profile in ("standard", "graph"):
        ctx = ToolContext(
            working_dir=str(tmp_path), permission_mode="bench", use_sandbox=False,
            tool_profile=profile,
        )
        res = await dispatch("explore", {"query": "zzz_nothing_like_this"}, ctx)
        text = res.data["result"]
        assert "No symbols match" in text
        if profile_shows(profile, "search_files"):
            assert "search_files" in text
        else:
            assert "search_files" not in text
            assert "run_command" in text, "it still has to name a way forward"
