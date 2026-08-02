"""Code knowledge graph: symbols plus the edges between them.

The `index` package answers "what is in this repo" — a tree of files and
symbols with summaries, searched lexically by `recall`. This package answers
the questions a tree cannot: who calls this, what breaks if I change it, what
does this class inherit from. Nodes are files/classes/functions/methods,
edges are contains/imports/calls/inherits, and both directions are stored so
the reverse queries are lookups rather than scans.

Persisted next to the index in `.squishy/graph.json` and built by the same
`/init` that builds the index, so there is one command to run and one
directory to gitignore.
"""

from __future__ import annotations

import os
from pathlib import Path

from squishy.graph.builder import build_graph
from squishy.graph.models import Edge, EdgeKind, Node, NodeKind
from squishy.graph.store import CodeGraph

GRAPH_FILE = "graph.json"


def graph_path(cwd: str | os.PathLike[str]) -> Path:
    from squishy.index.store import index_dir

    return index_dir(cwd) / GRAPH_FILE


def has_graph(cwd: str | os.PathLike[str]) -> bool:
    """True only if a *loadable* graph exists.

    Mirrors ``index.store.has_index``: a truncated ``graph.json`` from an
    interrupted ``/init`` must not make the harness advertise `explore`, or
    the model is pointed at a tool that can only ever error.
    """
    path = graph_path(cwd)
    if not path.is_file():
        return False
    try:
        CodeGraph.load(path)
    except Exception:  # noqa: BLE001
        return False
    return True


def load_graph(cwd: str | os.PathLike[str]) -> CodeGraph | None:
    path = graph_path(cwd)
    if not path.is_file():
        return None
    try:
        return CodeGraph.load(path)
    except Exception:  # noqa: BLE001
        return None


def save_graph(cwd: str | os.PathLike[str], graph: CodeGraph) -> Path:
    path = graph_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written through a temp file for the same reason the index is: readers
    # must see the old graph or the whole new one, never a half-written one.
    tmp = path.with_name(path.name + ".tmp")
    graph.save(tmp)
    os.replace(tmp, path)
    return path


def build_repo_graph(cwd: str | os.PathLike[str]) -> CodeGraph:
    """Build the graph for a working directory and persist it."""
    graph = build_graph(Path(cwd))
    save_graph(cwd, graph)
    return graph


__all__ = [
    "CodeGraph",
    "Edge",
    "EdgeKind",
    "GRAPH_FILE",
    "Node",
    "NodeKind",
    "build_graph",
    "build_repo_graph",
    "graph_path",
    "has_graph",
    "load_graph",
    "save_graph",
]
