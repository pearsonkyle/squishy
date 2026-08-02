"""The graph's read-only queries — the answers `explore` is built out of."""

from __future__ import annotations

from pathlib import Path

from squishy.graph.builder import build_graph
from squishy.graph.query import (
    explore,
    file_outline,
    impact_of,
    no_match_guidance,
    repo_map,
)


def test_repo_map_names_every_file(sample_repo: Path) -> None:
    out = repo_map(build_graph(sample_repo))
    assert "demo/services.py" in out
    assert "demo/utils.py" in out


def test_explore_returns_source_callers_and_impact(sample_repo: Path) -> None:
    out = explore(build_graph(sample_repo), sample_repo, "slugify")
    assert "def slugify" in out          # verbatim source
    assert "UserService.normalize" in out  # its caller
    assert "Impact" in out


def test_impact_lists_transitive_dependents(sample_repo: Path) -> None:
    out = impact_of(build_graph(sample_repo), "slugify", depth=2)
    assert "UserService.greet" in out


def test_a_miss_names_the_nearest_symbols(sample_repo: Path) -> None:
    """A dead end costs a turn; the closest names cost nothing and are callable.

    Observed on qiskit-terra-5662: told only to "try a shorter substring", the
    model ran the same failing query three times and then stopped.
    """
    graph = build_graph(sample_repo)
    out = no_match_guidance(graph, "slugfy")
    assert "slugify" in out
    assert "No symbols match" in out


def test_a_miss_with_nothing_close_says_so(sample_repo: Path) -> None:
    out = explore(build_graph(sample_repo), sample_repo, "zzz_not_a_symbol")
    assert "No symbols match" in out
    assert "search_files" in out, "must name a tool that exists"


def test_explore_elides_long_bodies_and_names_the_way_out(sample_repo: Path) -> None:
    """A 400-line body inside an explore result is paid for on every later turn.

    So orientation gets a capped view plus a pointer to the full one — and the
    pointer has to be a call the model can actually make, with its arguments
    already filled in.
    """
    long_body = "\n".join(f"    x{i} = {i}" for i in range(200))
    (sample_repo / "demo" / "big.py").write_text(
        f"def enormous():\n{long_body}\n    return x0\n", encoding="utf-8"
    )
    graph = build_graph(sample_repo)
    out = explore(graph, sample_repo, "enormous")
    assert "lines elided" in out
    assert "read_file('demo/big.py'" in out
    assert "offset=" in out and "limit=" in out
    assert len(out.splitlines()) < 120

    full = explore(graph, sample_repo, "enormous", max_source_lines=None)
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
    out = explore(graph, sample_repo, "slugify")
    assert "::slugify (" in out
    assert "slugify_all" not in out
    # With no exact match the fuzzy hits are all there is — keep them.
    assert "slugify_all" in explore(graph, sample_repo, "slugify_")


def test_outline_is_empty_for_a_file_the_graph_does_not_cover(
    sample_repo: Path,
) -> None:
    """Empty, not an error: the caller must fall back to a real read.

    Refusing a file the index happens not to cover would be the harness
    blocking an action it had just recommended.
    """
    graph = build_graph(sample_repo)
    assert file_outline(graph, "README.md") == ""
    assert file_outline(graph, "demo/does_not_exist.py") == ""


def test_outline_lists_spans_and_nested_methods(sample_repo: Path) -> None:
    out = file_outline(build_graph(sample_repo), "demo/services.py")
    assert "class UserService" in out
    assert "def greet" in out
    assert "L" in out, "line spans are the point — they make a follow-up read exact"
