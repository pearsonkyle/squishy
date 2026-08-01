"""Read-only queries over the code graph, as plain functions.

Kept free of tool plumbing so every answer here is unit-testable without a
model, a context, or a registry. `squishy/tools/graph.py` wraps these.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from squishy.graph.models import EdgeKind, Node, NodeKind
from squishy.graph.store import CodeGraph

# `explore` returns several symbols at once, and a single 400-line class body
# triples the cost of every subsequent turn — the transcript carries it
# forever. Orientation needs the signature, the docstring and the shape;
# verbatim text for an edit is what `symbol_source` is for.
EXPLORE_MAX_SOURCE_LINES = 60

# Exact matches score 3.0 minus a small length tiebreak, prefix matches 2.0.
_EXACT_MATCH_SCORE = 2.5

_IMPACT_PREVIEW = 15


def source_of(root: Path, node: Node, max_lines: int | None = None) -> str:
    """Line-numbered source of a node's span, optionally elided in the middle."""
    try:
        lines = (root / node.path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        return f"(source unavailable: {exc})"
    span = lines[node.lineno - 1 : node.end_lineno]
    numbered = [f"{i}\t{line}" for i, line in enumerate(span, start=node.lineno)]
    if max_lines is None or len(numbered) <= max_lines:
        return "\n".join(numbered)
    head, tail = max_lines * 3 // 4, max_lines // 4
    elided = len(numbered) - head - tail
    # The marker names a call the model can actually make, with the arguments
    # already filled in. "Elided" on its own costs a turn of guessing.
    resume = node.lineno + head - 1
    return "\n".join(
        numbered[:head]
        + [
            f"\t… {elided} lines elided — read_file('{node.path}', "
            f"offset={resume}, limit={elided + tail}) for the full body …"
        ]
        + numbered[-tail:]
    )


def file_outline(graph: CodeGraph, relative: str) -> str:
    """Symbol map of one indexed file: line spans, kinds, first docstring line.

    Returned instead of a file body only when the body would be truncated
    anyway (see `read_file`). Empty string when the file is not in the graph,
    so the caller falls back to a real read: refusing a file the index does
    not cover would be the harness blocking what it just instructed.
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
        "Outline only — the full file is too long to return in one call. For a "
        "body call explore('name'), or read_file with offset/limit for exact lines."
    )
    return "\n".join(lines)


def impact_of(graph: CodeGraph, symbol: str, depth: int = 2) -> str:
    """Blast radius: everything that transitively depends on ``symbol``."""
    hits = graph.search(symbol, limit=1)
    if not hits:
        return no_match_guidance(graph, symbol)
    seed = hits[0]
    impacted = graph.impact(seed.node_id, depth=depth)
    if not impacted:
        return (
            f"Nothing in the graph depends on {seed.node_id}. It is safe to "
            "change in isolation."
        )
    lines = [f"Impact of changing {seed.node_id} (depth {depth}):"]
    lines += [f"  {n.node_id} ({n.kind.value}, L{n.lineno})" for n in impacted]
    return "\n".join(lines)


def no_match_guidance(graph: CodeGraph, query: str) -> str:
    """A miss that names the nearest symbols, not a dead end.

    "Try a shorter substring" is advice the model cannot act on without
    another round trip, so on qiskit-terra-5662 it simply ran the same failing
    query three times. The graph already knows every name in the repo; the
    closest ones cost nothing to include and are directly callable.
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
            + "\nCall explore on one of those, or search_files for the string if "
            "it is not a Python symbol."
        )
    return (
        f"No symbols match {query!r}, and nothing in the graph is close to it. "
        "It may not be a Python symbol — search_files for the literal string, "
        "or call repo_map for an overview."
    )


def explore(
    graph: CodeGraph,
    root: Path,
    query: str,
    limit: int = 3,
    max_source_lines: int | None = EXPLORE_MAX_SOURCE_LINES,
) -> str:
    """One call: matching symbols' source, their callers/callees, their impact.

    Designed so a single call replaces a search/read crawl.
    """
    scored = graph.search_scored(query, limit=limit)
    if not scored:
        return no_match_guidance(graph, query)
    # An exact hit answers the question; the substring matches underneath it
    # are noise that stays in the transcript for the rest of the run.
    if scored[0][0] >= _EXACT_MATCH_SCORE:
        scored = [pair for pair in scored if pair[0] >= _EXACT_MATCH_SCORE]
    sections: list[str] = []
    for _score, node in scored:
        if node.kind is NodeKind.FILE:
            children = ", ".join(c.name for c in graph.children(node.node_id))
            sections.append(f"## {node.path} (file)\nContains: {children}")
            continue
        block = [
            f"## {node.node_id} ({node.kind.value})",
            "```python",
            source_of(root, node, max_source_lines),
            "```",
        ]
        callers = graph.callers(node.node_id)
        callees = graph.callees(node.node_id)
        subclasses = graph.reverse_neighbors(node.node_id, EdgeKind.INHERITS)
        if callers:
            block.append("Called by: " + ", ".join(c.node_id for c in callers))
        if callees:
            block.append("Calls: " + ", ".join(c.node_id for c in callees))
        if subclasses:
            block.append("Subclassed by: " + ", ".join(s.node_id for s in subclasses))
        impacted = graph.impact(node.node_id, depth=2)
        if impacted:
            block.append(
                "Impact radius (depth 2): "
                + ", ".join(n.node_id for n in impacted[:_IMPACT_PREVIEW])
            )
        sections.append("\n".join(block))
    return "\n\n".join(sections)


def repo_map(graph: CodeGraph, max_chars: int = 8000) -> str:
    """Compact overview of every file with its classes and functions."""
    return graph.repo_map(max_chars=max_chars)


__all__ = [
    "EXPLORE_MAX_SOURCE_LINES",
    "explore",
    "file_outline",
    "impact_of",
    "no_match_guidance",
    "repo_map",
    "source_of",
]
