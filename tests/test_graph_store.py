"""Tests for CodeGraph queries, persistence, and thread safety."""

from __future__ import annotations

import threading
from pathlib import Path

from squishy.graph.builder import build_graph
from squishy.graph.models import EdgeKind, Node, NodeKind
from squishy.graph.store import CodeGraph


def test_search_ranks_exact_name_first(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    results = graph.search("slugify")
    assert results
    assert results[0].node_id == "demo/utils.py::slugify"


def test_search_is_case_insensitive_substring(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    ids = {n.node_id for n in graph.search("userserv")}
    assert "demo/services.py::UserService" in ids


def test_impact_radius_walks_reverse_edges(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    impacted = {n.node_id for n in graph.impact("demo/utils.py::slugify", depth=2)}
    # normalize calls slugify; greet calls normalize -> both impacted.
    assert "demo/services.py::UserService.normalize" in impacted
    assert "demo/services.py::UserService.greet" in impacted
    # The seed symbol itself is not part of its own impact set.
    assert "demo/utils.py::slugify" not in impacted


def test_impact_depth_limits_traversal(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    impacted = {n.node_id for n in graph.impact("demo/utils.py::slugify", depth=1)}
    assert "demo/services.py::UserService.normalize" in impacted
    assert "demo/services.py::UserService.greet" not in impacted


def test_repo_map_lists_files_and_symbols(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    text = graph.repo_map()
    assert "demo/services.py" in text
    assert "UserService" in text
    assert "slugify" in text


def test_round_trip_persistence(sample_repo: Path, tmp_path: Path) -> None:
    graph = build_graph(sample_repo)
    out = tmp_path / "graph.json"
    graph.save(out)
    loaded = CodeGraph.load(out)
    assert loaded.get("demo/utils.py::slugify") is not None
    callers = {c.node_id for c in loaded.callers("demo/utils.py::slugify")}
    assert "demo/app.py::main" in callers


def test_concurrent_writes_are_thread_safe() -> None:
    graph = CodeGraph()

    def add_many(offset: int) -> None:
        for i in range(200):
            node_id = f"f{offset}_{i}.py"
            graph.add_node(
                Node(
                    node_id=node_id,
                    kind=NodeKind.FILE,
                    name=node_id,
                    path=node_id,
                    lineno=1,
                    end_lineno=1,
                )
            )
            if i:
                graph.add_edge(f"f{offset}_{i - 1}.py", node_id, EdgeKind.IMPORTS)

    threads = [threading.Thread(target=add_many, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(graph.nodes_of_kind(NodeKind.FILE)) == 8 * 200


def test_repo_map_degrades_to_one_line_per_file_instead_of_truncating(
    sample_repo: Path,
) -> None:
    """Truncation is path-sorted, so it used to hide most of the repo behind an
    exhaustive tour of whatever sorted first."""
    graph = build_graph(sample_repo)
    tight = graph.repo_map(max_chars=200)
    assert "demo/utils.py" in tight
    assert "demo/services.py" in tight
    assert "… (truncated)" not in tight
    # Detail is what got dropped, not files.
    assert "def " not in tight
    assert len(tight) <= 200
