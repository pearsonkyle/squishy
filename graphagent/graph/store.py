"""In-memory, thread-safe code knowledge graph with JSON persistence.

All mutation and traversal of shared state is guarded by a single
``threading.Lock`` so the store can be shared between an indexing thread
(or file watcher) and agent tool calls.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from graphagent.graph.models import Edge, EdgeKind, Node, NodeKind


class CodeGraph:
    """Symbol/file graph supporting the queries agents need.

    Forward edges answer "what does X use?"; reverse edges answer
    "who uses X?" (callers, importers, subclasses, impact radius).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, Node] = {}
        self._fwd: dict[EdgeKind, dict[str, list[str]]] = {
            kind: defaultdict(list) for kind in EdgeKind
        }
        self._rev: dict[EdgeKind, dict[str, list[str]]] = {
            kind: defaultdict(list) for kind in EdgeKind
        }

    # ------------------------------------------------------------ mutation

    def add_node(self, node: Node) -> None:
        with self._lock:
            self._nodes[node.node_id] = node

    def add_edge(self, src: str, dst: str, kind: EdgeKind) -> None:
        with self._lock:
            if dst not in self._fwd[kind][src]:
                self._fwd[kind][src].append(dst)
            if src not in self._rev[kind][dst]:
                self._rev[kind][dst].append(src)

    # ------------------------------------------------------------- lookups

    def get(self, node_id: str) -> Node | None:
        with self._lock:
            return self._nodes.get(node_id)

    def nodes_of_kind(self, kind: NodeKind) -> list[Node]:
        with self._lock:
            return [n for n in self._nodes.values() if n.kind is kind]

    def neighbors(self, node_id: str, kind: EdgeKind) -> list[Node]:
        """Forward neighbors of ``node_id`` along ``kind`` edges."""
        with self._lock:
            ids = list(self._fwd[kind].get(node_id, ()))
            return [self._nodes[i] for i in ids if i in self._nodes]

    def reverse_neighbors(self, node_id: str, kind: EdgeKind) -> list[Node]:
        """Reverse neighbors of ``node_id`` along ``kind`` edges."""
        with self._lock:
            ids = list(self._rev[kind].get(node_id, ()))
            return [self._nodes[i] for i in ids if i in self._nodes]

    def children(self, node_id: str) -> list[Node]:
        """Symbols directly contained by a file or class."""
        return self.neighbors(node_id, EdgeKind.CONTAINS)

    def callers(self, node_id: str) -> list[Node]:
        return self.reverse_neighbors(node_id, EdgeKind.CALLS)

    def callees(self, node_id: str) -> list[Node]:
        return self.neighbors(node_id, EdgeKind.CALLS)

    # -------------------------------------------------------------- search

    def all_names(self) -> list[str]:
        """Every symbol name in the graph, sorted — for did-you-mean lookups."""
        with self._lock:
            return sorted({node.name for node in self._nodes.values()})

    def search(self, query: str, limit: int = 10) -> list[Node]:
        """Best matches for ``query``, most specific first."""
        return [node for _, node in self.search_scored(query, limit)]

    def search_scored(self, query: str, limit: int = 10) -> list[tuple[float, Node]]:
        """Case-insensitive substring search, with each hit's score.

        Exact name matches rank first, then prefix matches, then substring
        matches; ties break on shorter names (more specific symbols).

        Callers need the score, not just the order: when the query names a
        symbol exactly there is nothing to be gained by also returning the
        three functions that merely contain it as a substring, and in a tool
        result that noise is paid for on every subsequent turn.
        """
        q = query.strip().lower()
        if not q:
            return []
        scored: list[tuple[float, Node]] = []
        with self._lock:
            candidates = list(self._nodes.values())
        for node in candidates:
            name = node.name.lower()
            qual = node.node_id.lower()
            if name == q or qual.endswith(f"::{q}") or qual.endswith(f".{q}"):
                score = 3.0
            elif name.startswith(q):
                score = 2.0
            elif q in name or q in qual:
                score = 1.0
            else:
                continue
            scored.append((score - len(name) / 1000.0, node))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    # -------------------------------------------------------------- impact

    def impact(self, node_id: str, depth: int = 2) -> list[Node]:
        """Breadth-first walk of reverse call/import/inherit edges.

        Returns every symbol or file that (transitively, up to ``depth``
        hops) depends on ``node_id`` — the blast radius of changing it.
        The seed node itself is excluded.
        """
        seen: set[str] = {node_id}
        frontier: list[str] = [node_id]
        impacted: list[Node] = []
        for _ in range(max(depth, 0)):
            next_frontier: list[str] = []
            for current in frontier:
                for kind in (EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS):
                    for node in self.reverse_neighbors(current, kind):
                        if node.node_id in seen:
                            continue
                        seen.add(node.node_id)
                        impacted.append(node)
                        next_frontier.append(node.node_id)
            frontier = next_frontier
            if not frontier:
                break
        return impacted

    # ------------------------------------------------------------ repo map

    def repo_map(self, max_chars: int = 8000) -> str:
        """Compact, hierarchical overview of every file and its symbols.

        Above ``max_chars`` this drops to one line per file rather than
        truncating. Truncation is worse than it looks: the listing is sorted
        by path, so on a 140-file repo the model received an exhaustive tour
        of everything alphabetically before ``s`` and no evidence that the
        rest of the repo existed — an overview that hides most of the repo is
        not an overview.
        """
        for candidate in (
            self._repo_map_detailed(),
            self._repo_map_by_file(names=True),
            self._repo_map_by_file(names=False),
        ):
            if len(candidate) <= max_chars:
                return candidate
        return self._repo_map_by_file(names=False)[:max_chars] + "\n… (truncated)"

    def _repo_map_by_file(self, names: bool = True) -> str:
        """One line per file: path, symbol count, and optionally symbol names."""
        lines = ["(one line per file — call explore on anything that looks relevant)"]
        for file_node in sorted(self.nodes_of_kind(NodeKind.FILE), key=lambda n: n.path):
            children = [c.name for c in self.children(file_node.node_id)]
            entry = f"{file_node.path} ({len(children)})"
            if names and children:
                head = ", ".join(children[:5]) + ("…" if len(children) > 5 else "")
                entry = f"{entry}: {head}"
            lines.append(entry)
        return "\n".join(lines)

    def _repo_map_detailed(self) -> str:
        lines: list[str] = []
        files = sorted(self.nodes_of_kind(NodeKind.FILE), key=lambda n: n.path)
        for file_node in files:
            lines.append(file_node.path)
            children = sorted(
                self.children(file_node.node_id), key=lambda n: n.lineno
            )
            for child in children:
                doc = (
                    f"  # {child.docstring.splitlines()[0]}"
                    if child.docstring
                    else ""
                )
                lines.append(
                    f"  {child.kind.value} {child.name} (L{child.lineno}){doc}"
                )
                if child.kind is NodeKind.CLASS:
                    for method in sorted(
                        self.children(child.node_id), key=lambda n: n.lineno
                    ):
                        lines.append(
                            f"    {method.kind.value} {method.name}"
                            f" (L{method.lineno})"
                        )
        return "\n".join(lines)

    # -------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        with self._lock:
            payload = {
                "nodes": [n.to_dict() for n in self._nodes.values()],
                "edges": [
                    {"src": src, "dst": dst, "kind": kind.value}
                    for kind, adj in self._fwd.items()
                    for src, dsts in adj.items()
                    for dst in dsts
                ],
            }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> CodeGraph:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        graph = cls()
        for node_data in payload["nodes"]:
            graph.add_node(Node.from_dict(node_data))
        for edge_data in payload["edges"]:
            edge = Edge.from_dict(edge_data)
            graph.add_edge(edge.src, edge.dst, edge.kind)
        return graph

    # --------------------------------------------------------------- misc

    def stats(self) -> dict[str, int]:
        with self._lock:
            edge_count = sum(
                len(dsts) for adj in self._fwd.values() for dsts in adj.values()
            )
            return {"nodes": len(self._nodes), "edges": edge_count}

    def all_symbol_names(self) -> Iterable[str]:
        with self._lock:
            return [n.name for n in self._nodes.values()]
