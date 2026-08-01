"""Graph tools: `explore`, `impact_of`, `repo_map`.

`recall` searches the index and answers "where does this live". These answer
the follow-up questions that cost a crawl: what is the code, who calls it,
and what else moves if I change it. `explore` is deliberately one strong tool
rather than three weak ones — a single call returns the matching symbols'
source, their callers and callees, their subclasses and their impact radius,
which is the whole first phase of a bug fix in one round trip.

Every result here is text rather than a JSON tree. It is read by a model, and
source code survives a JSON round trip badly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from squishy.graph import graph_path, load_graph
from squishy.graph.query import explore as _explore
from squishy.graph.query import impact_of as _impact_of
from squishy.graph.query import repo_map as _repo_map
from squishy.graph.store import CodeGraph
from squishy.tool_restrictions import profile_shows
from squishy.tools.base import Tool, ToolContext, ToolResult

DEFAULT_EXPLORE_LIMIT = 3
MAX_EXPLORE_LIMIT = 8
DEFAULT_IMPACT_DEPTH = 2
MAX_IMPACT_DEPTH = 4

_NO_GRAPH = (
    "no code graph found. Run /init to build .squishy/graph.json, or find the "
    "code by reading and searching directly."
)


def graph_for(ctx: ToolContext) -> CodeGraph | None:
    """Load the graph once per session, reloading when /init rewrites it.

    Same contract as `recall`'s index cache: hold it for the session, but
    invalidate on mtime so a mid-session rebuild is picked up rather than
    serving a stale graph for the rest of the run.
    """
    try:
        mtime = graph_path(ctx.working_dir).stat().st_mtime
    except OSError:
        mtime = -1.0
    if ctx._cached_graph is None or mtime != ctx._cached_graph_mtime:
        ctx._cached_graph = load_graph(ctx.working_dir)
        ctx._cached_graph_mtime = mtime
    graph: CodeGraph | None = ctx._cached_graph
    return graph


def _clip(value: Any, limit: int) -> int:
    """Coerce a model-supplied bound into a sane range."""
    return max(1, min(int(value), limit))


async def _explore_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = args.get("query") or args.get("symbol") or args.get("name")
    if not isinstance(query, str) or not query.strip():
        return ToolResult(False, error="`query` is required (string)")
    graph = graph_for(ctx)
    if graph is None:
        return ToolResult(False, error=_NO_GRAPH)
    try:
        limit = _clip(args.get("limit", DEFAULT_EXPLORE_LIMIT), MAX_EXPLORE_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_EXPLORE_LIMIT
    text = _explore(
        graph, Path(ctx.working_dir), query.strip(), limit=limit,
        # A dead end that names an invisible tool is a dead end twice over.
        search_tool=(
            "search_files"
            if profile_shows(ctx.tool_profile, "search_files")
            else "`grep` via run_command"
        ),
    )
    return ToolResult(
        True,
        data={"query": query, "result": text},
        display=f"{text.count('## ') or 'no'} match(es)",
    )


async def _impact_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    symbol = args.get("symbol") or args.get("query") or args.get("name")
    if not isinstance(symbol, str) or not symbol.strip():
        return ToolResult(False, error="`symbol` is required (string)")
    graph = graph_for(ctx)
    if graph is None:
        return ToolResult(False, error=_NO_GRAPH)
    try:
        depth = _clip(args.get("depth", DEFAULT_IMPACT_DEPTH), MAX_IMPACT_DEPTH)
    except (TypeError, ValueError):
        depth = DEFAULT_IMPACT_DEPTH
    text = _impact_of(graph, symbol.strip(), depth=depth)
    # One dependent per line after the header, so the count is the line count.
    dependents = max(0, text.count("\n")) if text.startswith("Impact of") else 0
    return ToolResult(
        True,
        data={"symbol": symbol, "result": text},
        display=f"{dependents} dependent(s)" if dependents else "nothing depends on it",
    )


async def _repo_map_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    graph = graph_for(ctx)
    if graph is None:
        return ToolResult(False, error=_NO_GRAPH)
    # The repo map is an orientation device; it must never be the thing that
    # blows the turn's budget. Sized against this run's own output cap.
    text = _repo_map(graph, max_chars=max(2000, ctx.max_tool_output_chars // 2))
    return ToolResult(True, data={"result": text}, display=f"{text.count(chr(10))} lines")


explore = Tool(
    name="explore",
    description=(
        "Answer a code question in one call using the repo's knowledge graph: "
        "returns the matching symbols' source, their callers and callees, "
        "their subclasses, and what depends on them. Prefer this over reading "
        "files one at a time when you are looking for how something works."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A symbol name, a dotted name like ClassName.method, or a "
                    "keyword that appears in one."
                ),
            },
            "limit": {
                "type": "integer",
                "default": DEFAULT_EXPLORE_LIMIT,
                "description": "Maximum number of symbols to return.",
            },
        },
        "required": ["query"],
    },
    run=_explore_tool,
)

impact_of = Tool(
    name="impact_of",
    description=(
        "List everything that transitively depends on a symbol — its callers, "
        "importers and subclasses. Use before changing shared code to see what "
        "else you might break."
    ),
    parameters={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Symbol name to analyze."},
            "depth": {
                "type": "integer",
                "default": DEFAULT_IMPACT_DEPTH,
                "description": "How many reverse-dependency hops to follow.",
            },
        },
        "required": ["symbol"],
    },
    run=_impact_tool,
)

repo_map = Tool(
    name="repo_map",
    description=(
        "Compact overview of every file in the repo with its classes and "
        "functions. Use once for orientation in an unfamiliar codebase."
    ),
    parameters={"type": "object", "properties": {}},
    run=_repo_map_tool,
)

GRAPH_TOOLS: list[Tool] = [explore, impact_of, repo_map]

__all__ = ["GRAPH_TOOLS", "explore", "graph_for", "impact_of", "repo_map"]
