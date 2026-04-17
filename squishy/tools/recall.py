"""`recall` tool — lexical search over the repo index.
 
Scores tree nodes by (a) name+path substring match, (b) token overlap with
summary. Returns the top-K matching subtrees trimmed to `depth` children deep.
Enough for the model to decide what to `read_file` next.
"""
 
from __future__ import annotations
 
import re
from typing import Any
 
from squishy.index.model import Index, Node
from squishy.index.store import load_index
from squishy.tools.base import Tool, ToolContext, ToolResult
 
MAX_RESULTS = 25
DEFAULT_LIMIT = 10
DEFAULT_DEPTH = 2
 
_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")
 
 
def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RX.findall(s) if len(t) >= 2}
 
 
def _score(node: Node, q_lower: str, q_tokens: set[str]) -> float:
    name_l = node.name.lower()
    path_l = node.path.lower()
    summary_l = node.summary.lower()
 
    score = 0.0
    if q_lower and q_lower in name_l:
        score += 10.0 if name_l == q_lower else 6.0
    if q_lower and q_lower in path_l:
        score += 3.0
    if q_lower and q_lower in summary_l:
        score += 2.0
 
    name_tokens = _tokens(node.name)
    path_tokens = _tokens(node.path)
    summary_tokens = _tokens(node.summary)
 
    score += 4.0 * len(q_tokens & name_tokens)
    score += 1.5 * len(q_tokens & path_tokens)
    score += 1.0 * len(q_tokens & summary_tokens)
 
    # Small bonus for leaf symbols — only when the node matched at all.
    if score > 0 and node.kind in ("class", "function", "method"):
        score += 0.5
    return score
 
 
def _trim(node: Node, depth: int) -> dict[str, Any]:
    d: dict[str, Any] = {
        "kind": node.kind,
        "name": node.name,
        "path": node.path,
    }
    if node.start_line or node.end_line:
        d["lines"] = [node.start_line, node.end_line]
    if node.summary:
        d["summary"] = node.summary
    if depth > 0 and node.children:
        d["children"] = [_trim(c, depth - 1) for c in node.children[:8]]
    return d
 
 
async def _recall(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return ToolResult(False, error="`query` is required (string)")
 
    depth = int(args.get("depth", DEFAULT_DEPTH))
    limit = min(int(args.get("limit", DEFAULT_LIMIT)), MAX_RESULTS)
    depth = max(0, min(depth, 4))
 
    idx = load_index(ctx.working_dir)
    if idx is None:
        return ToolResult(
            False,
            error="no index found. Run /init first to build .squishy/index.json",
        )
 
    q_lower = query.strip().lower()
    q_tokens = _tokens(query)
    if not q_tokens and not q_lower:
        return ToolResult(False, error="query contains no searchable tokens")
 
    scored: list[tuple[float, Node]] = []
    for node in idx.root.walk():
        if node.kind == "repo":
            continue
        s = _score(node, q_lower, q_tokens)
        if s > 0:
            scored.append((s, node))
    scored.sort(key=lambda t: (-t[0], t[1].path, t[1].name))
 
    results = [_trim(n, depth) for _, n in scored[:limit]]
    return ToolResult(
        True,
        data={
            "query": query,
            "results": results,
            "total_matched": len(scored),
            "returned": len(results),
        },
        display=f"{len(results)} of {len(scored)} matches",
    )
 
 
recall = Tool(
    name="recall",
    description=(
        "Search the repo index for relevant files, dirs, or symbols. Returns ranked "
        "entries with path, kind, line range, and summary. Use this before "
        "list_directory or search_files when you need to locate the right module. "
        "Requires /init to have been run."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language phrase or symbol name",
            },
            "limit": {"type": "integer", "default": DEFAULT_LIMIT},
            "depth": {
                "type": "integer",
                "default": DEFAULT_DEPTH,
                "description": "How many child levels to include per result",
            },
        },
        "required": ["query"],
    },
    run=_recall,
    mutates=False,
)
 
RECALL_TOOLS: list[Tool] = [recall]
 
 
__all__ = ["recall", "RECALL_TOOLS"]