"""`recall` tool — lexical search over the repo index.
 
Scores tree nodes by (a) name+path substring match, (b) token overlap with
summary. Returns the top-K matching subtrees trimmed to `depth` children deep.
Enough for the model to decide what to `read_file` next.
"""
 
from __future__ import annotations
 
import json
import re
from typing import Any
 
from squishy.index.model import Node
from squishy.index.store import index_path, load_index
from squishy.tools.base import Tool, ToolContext, ToolResult
 
MAX_RESULTS = 25
DEFAULT_LIMIT = 10
DEFAULT_DEPTH = 2
# Results are capped by token budget as well as count, so a recall over big
# classes can't quietly cost thousands of tokens.
DEFAULT_TOKEN_BUDGET = 1200
 
_TOKEN_RX = re.compile(r"[A-Za-z0-9_]+")
# Split camelCase/PascalCase: "JSONQuery" → ["JSON", "Query"]
_CAMEL_RX = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Path segments that mark lower-priority (non-authored / non-source) code.
# A match under one of these is down-weighted so first-party source ranks
# above tests/examples/vendored code — the single most impactful lexical
# ranking signal. It is a *penalty*, never a filter: these still surface when
# they are the best (or only) match, so exploration is never blocked.
_DOWNRANK_SEGMENTS = frozenset({
    "tests", "test", "vendor", "third_party", "third-party",
    "examples", "example", "docs", "doc", "fixtures", "testdata",
    "site-packages", "node_modules",
})
_DOWNRANK_FACTOR = 0.55


def _is_downranked(path: str) -> bool:
    """True if any path segment marks non-authored / non-source code."""
    if not path:
        return False
    segs = path.lower().replace("\\", "/").split("/")
    if any(s in _DOWNRANK_SEGMENTS for s in segs):
        return True
    # test_*.py / *_test.go style filenames.
    base = segs[-1]
    return base.startswith("test_") or base.endswith(("_test.py", "_test.go"))


def _tokens(s: str) -> set[str]:
    """Extract searchable tokens, splitting snake_case and camelCase."""
    tokens: set[str] = set()
    for raw in _TOKEN_RX.findall(s):
        low = raw.lower()
        if len(low) >= 2:
            tokens.add(low)
        # Also split camelCase/PascalCase sub-words so "JSONQuery" yields
        # {"jsonquery", "json", "query"} and matches query "json_query".
        for part in _CAMEL_RX.findall(raw):
            if len(part) >= 2:
                tokens.add(part.lower())
    return tokens
 
 
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
    # Down-weight non-authored / non-source matches so first-party code ranks
    # higher, but never zero them out — they remain available.
    if score > 0 and _is_downranked(node.path):
        score *= _DOWNRANK_FACTOR
    return score
 
 
def _trim(
    node: Node, depth: int, q_lower: str = "", q_tokens: set[str] | None = None,
) -> dict[str, Any]:
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
        # Surface the children most relevant to the query first, so a matching
        # method in a >8-member class isn't dropped by arbitrary source order.
        qt = q_tokens or set()
        children = sorted(
            node.children,
            key=lambda c: (-_score(c, q_lower, qt), c.name),
        )
        d["children"] = [_trim(c, depth - 1, q_lower, qt) for c in children[:8]]
    return d
 
 
async def _recall(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return ToolResult(False, error="`query` is required (string)")
 
    depth = int(args.get("depth", DEFAULT_DEPTH))
    limit = min(int(args.get("limit", DEFAULT_LIMIT)), MAX_RESULTS)
    depth = max(0, min(depth, 4))
 
    # Cache the loaded index, but invalidate when index.json changes on disk
    # (e.g. after a `/init` rebuild mid-session) so recall never serves a
    # stale tree for the whole session.
    try:
        cur_mtime = index_path(ctx.working_dir).stat().st_mtime
    except OSError:
        cur_mtime = -1.0
    idx = ctx._cached_index
    if idx is None or cur_mtime != ctx._cached_index_mtime:
        idx = load_index(ctx.working_dir)
        ctx._cached_index = idx
        ctx._cached_index_mtime = cur_mtime
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

    budget = int(args.get("token_budget", DEFAULT_TOKEN_BUDGET))
    results: list[dict[str, Any]] = []
    used = 0
    for _s, node in scored[:limit]:
        entry = _trim(node, depth, q_lower, q_tokens)
        cost = len(json.dumps(entry, ensure_ascii=False)) // 3.5
        # Always emit the top hit: the whole point is to name the best file.
        if results and used + cost > budget:
            break
        results.append(entry)
        used += cost

    data: dict[str, Any] = {
        "query": query,
        "results": results,
        "total_matched": len(scored),
        "returned": len(results),
    }
    if len(results) < len(scored):
        # Announce truncation explicitly: silence reads to the model as
        # "there is nothing else", which is exactly when it stops looking.
        data["truncated"] = (
            f"showing {len(results)} of {len(scored)} matches (~{budget}-token "
            "budget). Narrow the query, raise token_budget, or use search_files "
            "for exhaustive matching."
        )
    return ToolResult(
        True,
        data=data,
        display=f"{len(results)} of {len(scored)} matches",
    )
 
 
def recall_from_index(
    workspace: str, problem_text: str, limit: int = 8,
) -> list[dict[str, Any]]:
    """Pre-query the repo index using a free-text problem statement.

    Returns a list of ``{kind, name, path, lines, summary}`` dicts for the
    most relevant symbols/files — used to seed the agent with likely-relevant
    code (bench prompt build + post-compaction re-injection). Lives here (not
    in the bench package) so the core loop never imports from ``squishy.bench``.

    When a recalled symbol is a class in a multi-class file, its sibling
    classes are also surfaced so the agent doesn't fix one and miss the rest.
    """
    from squishy.index.store import load_index

    idx = load_index(str(workspace))
    if idx is None:
        return []

    q_lower = problem_text.strip().lower()[:500]
    q_tokens = _tokens(problem_text)
    if not q_tokens:
        return []

    scored: list[tuple[float, Node]] = []
    for node in idx.root.walk():
        if node.kind == "repo":
            continue
        s = _score(node, q_lower, q_tokens)
        if s > 0:
            scored.append((s, node))
    scored.sort(key=lambda t: (-t[0], t[1].path, t[1].name))

    # Build a path → file-node map once so sibling-class lookup is O(1).
    file_by_path: dict[str, Node] = {}
    for node in idx.root.walk():
        if node.kind == "file":
            file_by_path[node.path] = node

    results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _emit(node: Node) -> bool:
        key = f"{node.path}:{node.name}"
        if key in seen_keys:
            return False
        seen_keys.add(key)
        results.append(_trim(node, 0))
        return True

    for _, node in scored[: limit * 3]:
        if len(results) >= limit:
            break
        if not _emit(node):
            continue
        if node.kind != "class":
            continue
        file_node = file_by_path.get(node.path)
        if file_node is None:
            continue
        for sib in file_node.children:
            if sib.kind != "class" or sib.name == node.name:
                continue
            if len(results) >= limit:
                break
            _emit(sib)

    return results


recall = Tool(
    name="recall",
    description=(
        "Find where something lives in this repo by searching a prebuilt index "
        "of files and symbols. Returns ranked paths with line ranges and "
        "summaries — use it to locate code before reading files. If it returns "
        "nothing useful, fall back to reading and searching directly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What you are looking for: a symbol name, or a phrase "
                    "describing the behavior, e.g. 'parses unary operators'."
                ),
            },
            "limit": {
                "type": "integer", "default": DEFAULT_LIMIT,
                "description": "Maximum number of results to return.",
            },
            "token_budget": {
                "type": "integer",
                "default": DEFAULT_TOKEN_BUDGET,
                "description": "Max tokens of results; raise if truncated",
            },
            "depth": {
                "type": "integer",
                "default": DEFAULT_DEPTH,
                "description": "How many child levels to include per result",
            },
        },
        "required": ["query"],
    },
    run=_recall,
)
 
RECALL_TOOLS: list[Tool] = [recall]
 
 
__all__ = ["recall", "recall_from_index", "RECALL_TOOLS"]