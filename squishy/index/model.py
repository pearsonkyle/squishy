"""Dataclasses for the repo index.
 
The tree mirrors the repository: `repo → dir → file → symbol`. Every node
carries a human-readable `summary` (docstring, header comment, or an LLM-generated
one). The index is JSON-serialisable so `.squishy/index.json` is grep-friendly
and diffable.
"""
 
from __future__ import annotations
 
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
 
NodeKind = str  # "repo" | "dir" | "file" | "class" | "function" | "symbol"
 
 
@dataclass
class Node:
    id: str
    kind: NodeKind
    name: str
    path: str  # posix-style relative to repo root; "" for root
    start_line: int = 0
    end_line: int = 0
    hash: str = ""  # blake2 of file contents (files only)
    summary: str = ""
    summary_hash: str = ""  # hash of children's summaries for dir caching
    last_summarized: float = 0.0  # unix timestamp
    children: list[Node] = field(default_factory=list)
 
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
        }
        if self.start_line or self.end_line:
            d["start_line"] = self.start_line
            d["end_line"] = self.end_line
        if self.hash:
            d["hash"] = self.hash
        if self.summary:
            d["summary"] = self.summary
        if self.summary_hash:
            d["summary_hash"] = self.summary_hash
        if self.last_summarized:
            d["last_summarized"] = self.last_summarized
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d
 
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        return cls(
            id=str(d["id"]),
            kind=str(d["kind"]),
            name=str(d["name"]),
            path=str(d.get("path", "")),
            start_line=int(d.get("start_line", 0)),
            end_line=int(d.get("end_line", 0)),
            hash=str(d.get("hash", "")),
            summary=str(d.get("summary", "")),
            summary_hash=str(d.get("summary_hash", "")),
            last_summarized=float(d.get("last_summarized", 0.0)),
            children=[cls.from_dict(c) for c in d.get("children", [])],
        )
 
    def walk(self) -> Iterator[Node]:
        yield self
        for child in self.children:
            yield from child.walk()
 
 
@dataclass
class IndexMeta:
    generated_at: float = 0.0  # unix timestamp
    file_hashes: dict[str, str] = field(default_factory=dict)  # path -> hash
    model: str = ""  # summarizer model id, empty if no summaries
    squishy_version: str = ""
    stats: dict[str, int] = field(default_factory=dict)  # {files, symbols, by_ext}
    summary_stats: dict[str, int] = field(default_factory=dict)  # {files_summarized, dirs_summarized}
 
    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "file_hashes": self.file_hashes,
            "model": self.model,
            "squishy_version": self.squishy_version,
            "stats": self.stats,
            "summary_stats": self.summary_stats,
        }
 
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IndexMeta:
        return cls(
            generated_at=float(d.get("generated_at", 0.0)),
            file_hashes=dict(d.get("file_hashes", {})),
            model=str(d.get("model", "")),
            squishy_version=str(d.get("squishy_version", "")),
            stats=dict(d.get("stats", {})),
            summary_stats=dict(d.get("summary_stats", {}) or {}),
        )
 
 
@dataclass
class Index:
    root: Node
    meta: IndexMeta = field(default_factory=IndexMeta)
 
    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root.to_dict(), "meta": self.meta.to_dict()}
 
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Index:
        return cls(
            root=Node.from_dict(d["root"]),
            meta=IndexMeta.from_dict(d.get("meta", {})),
        )
 
    def find_file(self, rel_path: str) -> Node | None:
        for node in self.root.walk():
            if node.kind == "file" and node.path == rel_path:
                return node
        return None
 
 
__all__ = ["Node", "NodeKind", "IndexMeta", "Index"]
