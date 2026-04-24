"""Assemble an `Index` from a repo walk + per-file symbol extraction.

Incremental: if a prior index exists, subtrees whose file hash is unchanged
are copied over (preserving their summaries). Only new/changed files pay the
AST+summary cost on subsequent runs.

Optimized with parallel pipeline stages:
  walk → hash (I/O bound) → parse (CPU bound) → summarize (network bound)
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from squishy.index import ast_generic, ast_py
from squishy.index.ast_py import Symbol
from squishy.index.model import Index, IndexMeta, Node
from squishy.index.walker import FileRecord, walk_repo

# Thread pool for parallel I/O-bound operations
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Get or create a shared thread pool for parallel operations."""
    global _executor
    if _executor is None:
        # More workers for I/O bound tasks
        _executor = ThreadPoolExecutor(
            max_workers=min(64, (os.cpu_count() or 4) * 2 + 8)
        )
    return _executor


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


def _file_node_from_source(rec: FileRecord, source: str) -> Node:
    """Build a file node from already-read source string."""
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


async def _process_file_record(
    rec: FileRecord, prior_files: dict[str, Node]
) -> tuple[Node | None, int]:
    """Process a single file record, returning (node, symbol_count).

    Reuses prior file node if hash matches. Uses thread pool for I/O.
    """
    # Check if we can reuse prior node
    reused = prior_files.get(rec.path)
    if reused is not None and reused.hash and reused.hash == rec.hash:
        symbol_count = sum(1 for n in reused.walk() if n.kind in ("class", "function", "method"))
        return reused, symbol_count

    # Read file in thread pool (I/O bound)
    loop = asyncio.get_running_loop()
    source = await loop.run_in_executor(_get_executor(), _read_text, rec.abs_path)

    # Process in thread pool (CPU bound - AST parsing)
    node = await loop.run_in_executor(_get_executor(), _file_node_from_source, rec, source)
    symbol_count = sum(1 for n in node.walk() if n.kind in ("class", "function", "method"))

    return node, symbol_count


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


async def _build_index_async(
    cwd: str | os.PathLike[str], *, prior: Index | None = None, concurrency: int = 8
) -> Index:
    """Async implementation of index building."""
    root_path = Path(cwd).resolve()
    
    # Stage 1: Walk repo (sequential)
    records, hit_cap = walk_repo(cwd)
    prior_files = _prior_file_map(prior)

    if not records:
        # Empty repo - return minimal index
        root = Node(id="repo:.", kind="repo", name=root_path.name or ".", path="")
        meta = IndexMeta(
            generated_at=time.time(),
            file_hashes={},
            squishy_version="",
            stats={"files": 0, "symbols": 0, "hit_cap": int(hit_cap)},
        )
        return Index(root=root, meta=meta)

    # Stage 2: Process files with bounded concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(rec: FileRecord) -> tuple[Node | None, int]:
        async with semaphore:
            return await _process_file_record(rec, prior_files)

    results = await asyncio.gather(*[_bounded(rec) for rec in records])

    # Collect results
    file_nodes: list[Node] = []
    by_ext: dict[str, int] = defaultdict(int)
    symbol_count = 0
    for rec, (node, count) in zip(records, results):
        by_ext[rec.ext or "<none>"] += 1
        if node is not None:
            file_nodes.append(node)
            symbol_count += count

    # Stage 3: Build directory tree (sequential, O(n))
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


def build_index(
    cwd: str | os.PathLike[str], *, prior: Index | None = None, concurrency: int = 8
) -> Index:
    """Walk `cwd`, extract symbols, and return an `Index`.

    Reuses file subtrees (with their summaries) from `prior` when the file
    hash is unchanged. Dir-level summaries are NOT reused — they're cheap to
    re-roll and would otherwise go stale when children change.

    Optimized with parallel pipeline stages:
      - walk_repo: sequential directory traversal
      - hash computation: I/O bound, parallel
      - AST parsing: CPU bound, parallel via thread pool
      - dir tree assembly: sequential

    Uses batched processing with bounded concurrency for memory efficiency.

    This is a sync function that wraps the async implementation.
    For async contexts, use `_build_index_async`.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running - use asyncio.run
        return asyncio.run(_build_index_async(cwd, prior=prior, concurrency=concurrency))

    # Event loop is running - run the async code in a separate thread
    import concurrent.futures

    def _run_in_thread():
        return asyncio.run(_build_index_async(cwd, prior=prior, concurrency=concurrency))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result()
