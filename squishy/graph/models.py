"""Data model for the code knowledge graph.

Nodes are identified by stable string ids:

* files:   ``"pkg/module.py"`` (repo-relative POSIX path)
* symbols: ``"pkg/module.py::ClassName.method_name"``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    """The kind of entity a node represents."""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeKind(str, Enum):
    """The kind of relationship an edge represents."""

    CONTAINS = "contains"
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"


@dataclass(frozen=True, slots=True)
class Node:
    """A single entity in the graph with its source location."""

    node_id: str
    kind: NodeKind
    name: str
    path: str
    lineno: int
    end_lineno: int
    docstring: str | None = None
    signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "name": self.name,
            "path": self.path,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "docstring": self.docstring,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        return cls(
            node_id=data["node_id"],
            kind=NodeKind(data["kind"]),
            name=data["name"],
            path=data["path"],
            lineno=data["lineno"],
            end_lineno=data["end_lineno"],
            docstring=data.get("docstring"),
            signature=data.get("signature"),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed, typed relationship between two nodes."""

    src: str
    dst: str
    kind: EdgeKind

    def to_dict(self) -> dict[str, str]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Edge:
        return cls(src=data["src"], dst=data["dst"], kind=EdgeKind(data["kind"]))


@dataclass(slots=True)
class SearchHit:
    """A ranked search result."""

    node: Node
    score: float = field(default=0.0)
