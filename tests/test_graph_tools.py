"""The graph tools as squishy sees them: registry, schema, dispatch, caching."""

from __future__ import annotations

from pathlib import Path

from squishy.graph import build_repo_graph, graph_path, has_graph, load_graph
from squishy.tools import REGISTRY, dispatch, openai_schemas
from squishy.tools.base import ToolContext


def _ctx(repo: Path) -> ToolContext:
    return ToolContext(working_dir=str(repo), permission_mode="yolo", use_sandbox=False)


async def test_explore_dispatches_and_answers(sample_repo: Path) -> None:
    build_repo_graph(sample_repo)
    res = await dispatch("explore", {"query": "slugify"}, _ctx(sample_repo))
    assert res.success
    assert "def slugify" in res.data["result"]


async def test_impact_of_dispatches(sample_repo: Path) -> None:
    build_repo_graph(sample_repo)
    res = await dispatch("impact_of", {"symbol": "slugify"}, _ctx(sample_repo))
    assert res.success
    assert "UserService" in res.data["result"]


async def test_repo_map_dispatches(sample_repo: Path) -> None:
    build_repo_graph(sample_repo)
    res = await dispatch("repo_map", {}, _ctx(sample_repo))
    assert res.success
    assert "demo/utils.py" in res.data["result"]


async def test_without_a_graph_the_error_names_the_fix(sample_repo: Path) -> None:
    res = await dispatch("explore", {"query": "slugify"}, _ctx(sample_repo))
    assert not res.success
    assert "/init" in res.error


def test_graph_tools_are_hidden_when_no_graph_exists() -> None:
    """Same rule as `recall`: never advertise a tool that can only error.

    A tool in the schema is a turn the model may spend to be told to run
    /init — which the schema could have communicated for free by omitting it.
    """
    names = {
        s["function"]["name"]
        for s in openai_schemas("bench", has_index=False, has_graph=False)
    }
    assert not names & {"explore", "impact_of", "repo_map"}
    with_graph = {
        s["function"]["name"]
        for s in openai_schemas("bench", has_index=False, has_graph=True)
    }
    assert {"explore", "impact_of", "repo_map"} <= with_graph


def test_graph_profile_offers_explore_but_not_the_rest() -> None:
    """A narrow profile exists to be narrow: only the tool that replaces a crawl."""
    names = {
        s["function"]["name"]
        for s in openai_schemas("bench", profile="graph", has_index=False)
    }
    assert "explore" in names
    assert "impact_of" not in names and "repo_map" not in names
    assert {"run_command", "read_file", "edit_file"} <= names


def test_graph_tools_are_read_only() -> None:
    """They must be usable in `edits` mode without an approval prompt."""
    from squishy.tools import check_permission

    for name in ("explore", "impact_of", "repo_map"):
        allowed, reason = check_permission(REGISTRY[name], "edits")
        assert allowed, f"{name}: {reason}"


async def test_the_graph_is_reloaded_after_a_rebuild(sample_repo: Path) -> None:
    """A mid-session /init must not leave the tools serving the old graph."""
    build_repo_graph(sample_repo)
    ctx = _ctx(sample_repo)
    first = await dispatch("explore", {"query": "brand_new_symbol"}, ctx)
    assert "No symbols match" in first.data["result"]

    (sample_repo / "demo" / "new.py").write_text(
        "def brand_new_symbol():\n    return 1\n", encoding="utf-8"
    )
    build_repo_graph(sample_repo)
    # Force a distinct mtime; the cache key is the file's mtime.
    import os
    p = graph_path(sample_repo)
    os.utime(p, (p.stat().st_atime + 10, p.stat().st_mtime + 10))

    second = await dispatch("explore", {"query": "brand_new_symbol"}, ctx)
    assert "def brand_new_symbol" in second.data["result"]


def test_a_truncated_graph_reads_as_absent(sample_repo: Path) -> None:
    """An interrupted /init must not make the harness advertise `explore`."""
    build_repo_graph(sample_repo)
    graph_path(sample_repo).write_text("{ not json", encoding="utf-8")
    assert not has_graph(sample_repo)
    assert load_graph(sample_repo) is None
