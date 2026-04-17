"""Assemble an `Index` from a repo walk + per-file symbol extraction.
 
Incremental: if a prior index exists, subtrees whose file hash is unchanged
are copied over (preserving their summaries). Only new/changed files pay the
AST+summary cost on subsequent runs.
"""
 
from __future__ import annotations
 
import os
import time
from collections import defaultdict
from pathlib import Path
 
from squishy.index import ast_generic, ast_py
from squishy.index.ast_py import Symbol
from squishy.index.model import Index, IndexMeta, Node
from squishy.index.walker import FileRecord, walk_repo
 
 
def _read_text(abs_path: str) -> str:
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""
 
 
def _first_line(s: str) -> str:
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""
 
 
def _symbol_to_node(sym: Symbol, parent_path: str, prefix: str) -> Node:
    node = Node(
        id=f"{prefix}:{sym.name}",
        kind=sym.kind,
        name=sym.name,
        path=parent_path,
        start_line=sym.start_line,
        end_line=sym.end_line,
        summary=_first_line(sym.docstring)[:200],
    )
    if sym.children:
        node.children = [_symbol_to_node(c, parent_path, node.id) for c in sym.children]
    return node
 
 
def _file_node(rec: FileRecord, source: str) -> Node:
    ext = rec.ext
    if ext in (".py", ".pyi"):
        summary = _first_line(ast_py.module_docstring(source))[:200] if source else ""
        symbols = ast_py.extract_symbols(source)
    else:
        summary = ast_generic.header_comment(source, ext)
        symbols = ast_generic.extract_symbols(source, ext)
 
    return Node(
        id=f"file:{rec.path}",
        kind="file",
        name=os.path.basename(rec.path) or rec.path,
        path=rec.path,
        hash=rec.hash,
        summary=summary,
        children=[_symbol_to_node(s, rec.path, f"file:{rec.path}") for s in symbols],
    )
 
 
def _build_dir_tree(file_nodes: list[Node], root_path: Path) -> Node:
    """Group file nodes under directory nodes that mirror the filesystem."""
    by_dir: dict[str, list[Node]] = defaultdict(list)
    for fn in file_nodes:
        d = os.path.dirname(fn.path)
        by_dir[d].append(fn)
 
    # Build directory nodes bottom-up. For simplicity, flatten into one pass:
    # for each unique directory, create a Node and attach files directly. We
    # then build the nesting by linking each dir to its parent.
    dir_nodes: dict[str, Node] = {}
    for d in sorted({os.path.dirname(fn.path) for fn in file_nodes}):
        dir_nodes[d] = Node(
            id=f"dir:{d}" if d else "dir:.",
            kind="dir",
            name=os.path.basename(d) if d else root_path.name or ".",
            path=d,
        )
 
    # Ensure every ancestor directory exists.
    for d in list(dir_nodes.keys()):
        parent = os.path.dirname(d)
        while parent and parent not in dir_nodes:
            dir_nodes[parent] = Node(
                id=f"dir:{parent}",
                kind="dir",
                name=os.path.basename(parent),
                path=parent,
            )
            parent = os.path.dirname(parent)
 
    # Attach files.
    for d, fns in by_dir.items():
        dir_nodes[d].children.extend(sorted(fns, key=lambda n: n.path))
 
    # Attach subdirs to their parents.
    roots: list[Node] = []
    for d, node in sorted(dir_nodes.items(), key=lambda kv: kv[0]):
        if d == "":
            roots.append(node)
            continue
        parent_dir = os.path.dirname(d)
        parent_node = dir_nodes.get(parent_dir)
        if parent_node is None:
            roots.append(node)
        else:
            parent_node.children.append(node)
 
    root = roots[0] if roots else Node(
        id="dir:.", kind="dir", name=root_path.name or ".", path="",
    )
 
    # Root gets re-labeled as repo.
    root.kind = "repo"
    root.id = "repo:."
    return root
 
 
def _prior_file_map(prior: Index | None) -> dict[str, Node]:
    if prior is None:
        return {}
    return {n.path: n for n in prior.root.walk() if n.kind == "file"}
 
 
def build_index(cwd: str | os.PathLike[str], *, prior: Index | None = None) -> Index:
    """Walk `cwd`, extract symbols, and return an `Index`.
 
    Reuses file subtrees (with their summaries) from `prior` when the file
    hash is unchanged. Dir-level summaries are NOT reused — they're cheap to
    re-roll and would otherwise go stale when children change.
    """
    root_path = Path(cwd).resolve()
    records, hit_cap = walk_repo(cwd)
    prior_files = _prior_file_map(prior)
 
    file_nodes: list[Node] = []
    by_ext: dict[str, int] = defaultdict(int)
    symbol_count = 0
    for rec in records:
        by_ext[rec.ext or "<none>"] += 1
        reused = prior_files.get(rec.path)
        if reused is not None and reused.hash and reused.hash == rec.hash:
            file_nodes.append(reused)
            symbol_count += sum(1 for n in reused.walk() if n.kind in ("class", "function", "method"))
            continue
        source = _read_text(rec.abs_path)
        fn = _file_node(rec, source)
        file_nodes.append(fn)
        symbol_count += sum(1 for n in fn.walk() if n.kind in ("class", "function", "method"))
 
    root = _build_dir_tree(file_nodes, root_path)
 
    meta = IndexMeta(
        generated_at=time.time(),
        file_hashes={rec.path: rec.hash for rec in records},
        squishy_version="",
        stats={
            "files": len(records),
            "symbols": symbol_count,
            "hit_cap": int(hit_cap),
            **{f"ext{k}": v for k, v in by_ext.items()},
        },
    )
    return Index(root=root, meta=meta)
 
 
__all__ = ["build_index"]
