"""Tests for the pure tool implementations used by both agent arms."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphagent.agentkit import tools
from graphagent.graph.builder import build_graph

# ---------------------------------------------------------------- baseline ---


def test_list_dir(sample_repo: Path) -> None:
    out = tools.list_dir(sample_repo, ".")
    assert "demo/" in out
    assert "README.md" in out


def test_list_dir_rejects_escape(sample_repo: Path) -> None:
    with pytest.raises(ValueError):
        tools.list_dir(sample_repo, "../..")


def test_read_file_slice(sample_repo: Path) -> None:
    out = tools.read_file(sample_repo, "demo/utils.py", start_line=4, end_line=6)
    assert "slugify" in out
    header, first_numbered = out.splitlines()[:2]
    assert header.startswith("demo/utils.py (lines 4-6")
    assert first_numbered.startswith("4")


def test_read_file_rejects_escape(sample_repo: Path) -> None:
    with pytest.raises(ValueError):
        tools.read_file(sample_repo, "../etc/passwd")


def test_grep_matches_with_line_numbers(sample_repo: Path) -> None:
    out = tools.grep(sample_repo, r"def slugify")
    assert "demo/utils.py" in out
    assert ":4:" in out


def test_grep_invalid_regex_is_reported(sample_repo: Path) -> None:
    out = tools.grep(sample_repo, "([")
    assert out.startswith("error:")


# ------------------------------------------------------------------- graph ---


def test_repo_map_tool(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    out = tools.repo_map(graph)
    assert "demo/services.py" in out


def test_explore_returns_source_callers_and_impact(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    out = tools.explore(graph, sample_repo, "slugify")
    assert "def slugify" in out  # verbatim source
    assert "UserService.normalize" in out  # caller
    assert "Impact" in out


def test_explore_unknown_symbol_gives_guidance(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    out = tools.explore(graph, sample_repo, "does_not_exist_anywhere")
    assert "No symbols" in out


def test_symbol_source_includes_span(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    out = tools.symbol_source(graph, sample_repo, "UserService.greet")
    assert "demo/services.py" in out
    assert "def greet" in out


def test_impact_tool(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    out = tools.impact_of(graph, "slugify", depth=2)
    assert "UserService.greet" in out


def test_explore_elides_long_bodies_and_names_the_way_out(sample_repo: Path) -> None:
    """A 400-line body inside an explore result is paid for on every later
    turn, so orientation gets a capped view and a pointer to the full one."""
    long_body = "\n".join(f"    x{i} = {i}" for i in range(200))
    (sample_repo / "demo" / "big.py").write_text(
        f"def enormous():\n{long_body}\n    return x0\n", encoding="utf-8"
    )
    graph = build_graph(sample_repo)
    out = tools.explore(graph, sample_repo, "enormous")
    assert "lines elided" in out
    assert "symbol_source('enormous')" in out
    assert len(out.splitlines()) < 120

    full = tools.explore(graph, sample_repo, "enormous", max_source_lines=None)
    assert "lines elided" not in full
    assert "x199 = 199" in full


def test_explore_drops_substring_noise_when_the_query_hits_exactly(
    sample_repo: Path,
) -> None:
    """`slugify` matches `slugify_all` too; returning both bodies costs every
    later turn for a symbol the model did not ask about."""
    (sample_repo / "demo" / "extra.py").write_text(
        "def slugify_all(items):\n    return [i for i in items]\n", encoding="utf-8"
    )
    graph = build_graph(sample_repo)
    assert {n.name for n in graph.search("slugify")} == {"slugify", "slugify_all"}
    out = tools.explore(graph, sample_repo, "slugify")
    assert "::slugify (" in out
    assert "slugify_all" not in out
    # With no exact match, the fuzzy hits are all there is — keep them.
    assert "slugify_all" in tools.explore(graph, sample_repo, "slugify_")
