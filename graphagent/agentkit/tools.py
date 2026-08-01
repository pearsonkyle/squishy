"""Pure tool implementations shared by both agent arms.

These are plain functions with full type hints so they can be unit tested
without an LLM. The agent factories wrap them with ``@function_tool``.

All filesystem access validates that the requested path stays inside the
repository root — tools never read outside the sandboxed repo.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from graphagent.graph.models import EdgeKind, Node, NodeKind
from graphagent.graph.store import CodeGraph

_MAX_GREP_MATCHES = 50
_MAX_READ_LINES = 400
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".rst", ".json"}


def _safe_resolve(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``, rejecting path escapes."""
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes repository root: {relative!r}")
    return candidate


# ---------------------------------------------------------------- baseline ---


def list_dir(root: Path, relative: str = ".") -> str:
    """List entries of a directory (directories get a trailing slash)."""
    target = _safe_resolve(root, relative)
    if not target.is_dir():
        return f"error: not a directory: {relative}"
    entries: list[str] = []
    for entry in sorted(target.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)
    return "\n".join(entries) or "(empty)"


def read_file(
    root: Path,
    relative: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read a file slice with line numbers (capped at 400 lines)."""
    target = _safe_resolve(root, relative)
    if not target.is_file():
        return f"error: no such file: {relative}"
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(start_line, 1)
    stop = min(end_line or len(lines), len(lines), start + _MAX_READ_LINES - 1)
    numbered = [f"{i}\t{lines[i - 1]}" for i in range(start, stop + 1)]
    header = f"{relative} (lines {start}-{stop} of {len(lines)})"
    return header + "\n" + "\n".join(numbered)


def grep(root: Path, pattern: str, glob: str = "*.py") -> str:
    """Regex search across the repo; returns ``path:line: text`` matches."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid regex: {exc}"
    matches: list[str] = []
    for path in sorted(root.rglob(glob)):
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        if path.suffix not in _TEXT_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                matches.append(f"{rel}:{lineno}: {line.strip()}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    matches.append("… (truncated)")
                    return "\n".join(matches)
    return "\n".join(matches) or "no matches"


# ------------------------------------------------------------------- graph ---


def repo_map(graph: CodeGraph) -> str:
    """Compact overview of every file with its classes and functions."""
    return graph.repo_map()


def _source_of(root: Path, node: Node, max_lines: int | None = None) -> str:
    """Line-numbered source of a node's span, optionally elided in the middle.

    ``explore`` returns several symbols at once, and a single 400-line class
    body triples the cost of every subsequent turn — the transcript carries it
    forever. Orientation needs the signature, the docstring and the shape;
    verbatim text for an edit is what ``symbol_source`` is for, so the elision
    marker names it.
    """
    lines = (root / node.path).read_text(encoding="utf-8").splitlines()
    span = lines[node.lineno - 1 : node.end_lineno]
    numbered = [f"{i}\t{line}" for i, line in enumerate(span, start=node.lineno)]
    if max_lines is None or len(numbered) <= max_lines:
        return "\n".join(numbered)
    head, tail = max_lines * 3 // 4, max_lines // 4
    symbol = node.node_id.split("::")[-1]
    elided = len(numbered) - head - tail
    return "\n".join(
        numbered[:head]
        + [f"\t… {elided} lines elided — symbol_source({symbol!r}) for the full body …"]
        + numbered[-tail:]
    )


def exceeds_read_cap(graph: CodeGraph, relative: str) -> bool:
    """True when one ``read_file`` call cannot show the whole file.

    The threshold that decides whether an outline is worth its round trip.
    Read from the graph rather than the disk so the check costs nothing.
    """
    node = graph.get(relative)
    return node is not None and node.kind is NodeKind.FILE and (
        node.end_lineno > _MAX_READ_LINES
    )


def file_outline(graph: CodeGraph, relative: str) -> str:
    """Symbol map of one indexed file: line spans, kinds, first docstring line.

    Returned instead of 400 verbatim lines when the graph arm asks for a whole
    Python file. Across three seeds of qiskit-terra-5662 the graph arm made 27
    ``read_file`` calls, 17 of them whole-file, and spent a median of 781k
    input tokens against the filesystem arm's 246k — it was doing exactly the
    blind crawl the graph exists to replace, and every one of those file dumps
    stays in the transcript for the rest of the run.

    Empty string when the file is not in the graph, so the caller falls back
    to a real read: refusing a file the index does not cover would be the
    instruct-then-block failure again.
    """
    node = graph.get(relative)
    if node is None or node.kind is not NodeKind.FILE:
        return ""
    lines = [f"{relative} — outline ({node.end_lineno} lines)"]
    if node.docstring:
        lines.append(f'  """{node.docstring.splitlines()[0]}"""')
    for child in sorted(graph.children(node.node_id), key=lambda n: n.lineno):
        doc = f"  # {child.docstring.splitlines()[0]}" if child.docstring else ""
        lines.append(
            f"  L{child.lineno}-{child.end_lineno}  {child.kind.value} "
            f"{child.signature or child.name}{doc}"
        )
        for method in sorted(graph.children(child.node_id), key=lambda n: n.lineno):
            lines.append(
                f"    L{method.lineno}-{method.end_lineno}  {method.kind.value} "
                f"{method.signature or method.name}"
            )
    lines.append(
        "Outline only. For a body call symbol_source('name'), or read_file "
        "with start_line/end_line for exact lines."
    )
    return "\n".join(lines)


def symbol_source(graph: CodeGraph, root: Path, symbol: str) -> str:
    """Verbatim, line-numbered source of the best-matching symbol."""
    hits = graph.search(symbol, limit=1)
    if not hits:
        return f"No symbols match {symbol!r}. Try repo_map or a broader query."
    node = hits[0]
    return f"{node.path}::{node.node_id.split('::')[-1]}\n{_source_of(root, node)}"


def impact_of(graph: CodeGraph, symbol: str, depth: int = 2) -> str:
    """Blast radius: everything that transitively depends on ``symbol``."""
    hits = graph.search(symbol, limit=1)
    if not hits:
        return f"No symbols match {symbol!r}."
    seed = hits[0]
    impacted = graph.impact(seed.node_id, depth=depth)
    if not impacted:
        return f"Nothing in the graph depends on {seed.node_id}."
    lines = [f"Impact of changing {seed.node_id} (depth {depth}):"]
    lines += [f"  {n.node_id} ({n.kind.value}, L{n.lineno})" for n in impacted]
    return "\n".join(lines)


_EXPLORE_MAX_SOURCE_LINES = 60
# Exact matches score 3.0 minus a small length tiebreak, prefix matches 2.0.
_EXACT_MATCH_SCORE = 2.5


def _no_match_guidance(graph: CodeGraph, query: str) -> str:
    """A miss that names the nearest symbols, not a dead end.

    "Call repo_map or try a shorter substring" is advice the model cannot act
    on without another round trip, so on qiskit-terra-5662 it simply ran the
    same failing query three times. The graph already knows every name in the
    repo; the closest ones cost nothing to include and are directly callable.
    """
    names = graph.all_names()
    close = difflib.get_close_matches(query, names, n=6, cutoff=0.6)
    if not close:
        head = query.split("_")[0]
        close = [n for n in names if head and head.lower() in n.lower()][:6]
    if close:
        return (
            f"No symbols match {query!r}. Closest names in the graph: "
            + ", ".join(close)
            + "\nCall explore on one of those, or grep for the string if it is "
            "not a Python symbol."
        )
    return (
        f"No symbols match {query!r}, and nothing in the graph is close to it. "
        "It may not be a Python symbol — grep for the literal string, or call "
        "repo_map for an overview."
    )


def explore(
    graph: CodeGraph,
    root: Path,
    query: str,
    limit: int = 3,
    max_source_lines: int | None = _EXPLORE_MAX_SOURCE_LINES,
) -> str:
    """One-call answer: matching symbols' source + callers/callees + impact.

    Mirrors CodeGraph's ``codegraph_explore`` — designed so that a single
    call replaces a grep/read crawl.
    """
    scored = graph.search_scored(query, limit=limit)
    if not scored:
        return _no_match_guidance(graph, query)
    # An exact hit answers the question; the substring matches underneath it
    # are noise that stays in the transcript for the rest of the run.
    if scored[0][0] >= _EXACT_MATCH_SCORE:
        scored = [pair for pair in scored if pair[0] >= _EXACT_MATCH_SCORE]
    hits = [node for _, node in scored]
    sections: list[str] = []
    for node in hits:
        if node.kind is NodeKind.FILE:
            children = ", ".join(c.name for c in graph.children(node.node_id))
            sections.append(f"## {node.path} (file)\nContains: {children}")
            continue
        block = [f"## {node.node_id} ({node.kind.value})"]
        block.append("```python")
        block.append(_source_of(root, node, max_source_lines))
        block.append("```")
        callers = graph.callers(node.node_id)
        callees = graph.callees(node.node_id)
        subclasses = graph.reverse_neighbors(node.node_id, EdgeKind.INHERITS)
        if callers:
            block.append("Called by: " + ", ".join(c.node_id for c in callers))
        if callees:
            block.append("Calls: " + ", ".join(c.node_id for c in callees))
        if subclasses:
            block.append(
                "Subclassed by: " + ", ".join(s.node_id for s in subclasses)
            )
        impacted = graph.impact(node.node_id, depth=2)
        if impacted:
            block.append(
                "Impact radius (depth 2): "
                + ", ".join(n.node_id for n in impacted[:15])
            )
        sections.append("\n".join(block))
    return "\n\n".join(sections)
