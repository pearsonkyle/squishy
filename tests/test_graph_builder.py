"""Tests for the AST-based graph builder."""

from __future__ import annotations

from pathlib import Path

from squishy.graph.builder import build_graph
from squishy.graph.models import EdgeKind, NodeKind


def test_indexes_all_python_files(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    files = {n.path for n in graph.nodes_of_kind(NodeKind.FILE)}
    assert files == {
        "demo/__init__.py",
        "demo/app.py",
        "demo/services.py",
        "demo/utils.py",
    }


def test_extracts_symbols_with_spans(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    node = graph.get("demo/services.py::UserService.greet")
    assert node is not None
    assert node.kind is NodeKind.METHOD
    assert node.lineno < node.end_lineno
    assert node.docstring == "Return a greeting for ``name``."


def test_contains_edges_link_file_to_symbols(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    children = graph.children("demo/utils.py")
    names = {c.name for c in children}
    assert {"slugify", "unused_helper"} <= names


def test_inheritance_edge_resolved(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    bases = graph.neighbors("demo/services.py::AdminService", EdgeKind.INHERITS)
    assert [b.node_id for b in bases] == ["demo/services.py::UserService"]


def test_import_edges_resolved_to_files(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    imports = {
        e.node_id for e in graph.neighbors("demo/app.py", EdgeKind.IMPORTS)
    }
    assert "demo/services.py" in imports
    assert "demo/utils.py" in imports


def test_call_edges_cross_file(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    callees = {
        e.node_id
        for e in graph.neighbors(
            "demo/services.py::UserService.normalize", EdgeKind.CALLS
        )
    }
    assert "demo/utils.py::slugify" in callees


def test_callers_reverse_lookup(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    callers = {c.node_id for c in graph.callers("demo/utils.py::slugify")}
    assert "demo/services.py::UserService.normalize" in callers
    assert "demo/app.py::main" in callers


def test_syntax_errors_are_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    graph = build_graph(tmp_path)
    files = {n.path for n in graph.nodes_of_kind(NodeKind.FILE)}
    assert files == {"good.py"}


def test_file_nodes_carry_the_file_length(sample_repo: Path) -> None:
    """It was pinned at 1, so every file looked one line long. Harmless until
    something had to ask whether a file fits in a single read."""
    graph = build_graph(sample_repo)
    node = graph.get("demo/services.py")
    assert node is not None
    expected = len((sample_repo / "demo/services.py").read_text().splitlines())
    assert node.end_lineno == expected
