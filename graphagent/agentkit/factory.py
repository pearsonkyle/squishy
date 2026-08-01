"""Build the two agent arms with the OpenAI Agents SDK.

* **baseline** — only filesystem tools (list/read/grep): the agent must
  discover structure the slow way.
* **graph** — the pre-built knowledge graph tools (``explore`` first),
  with ``read_file`` kept as an escape hatch for content the graph
  doesn't carry (configs, docs).

Both arms share the same model and the same task framing so the only
variable in the A/B comparison is the toolset.
"""

from __future__ import annotations

from pathlib import Path

from agents import Agent, function_tool
from agents.models.interface import Model

from graphagent.agentkit import tools as impl
from graphagent.agentkit.llm import DEFAULT_MODEL
from graphagent.graph.store import CodeGraph

_BASELINE_INSTRUCTIONS = """\
You are a senior software engineer working inside a repository.
Explore with list_dir, grep, and read_file to find the code relevant to
the task, then answer precisely, citing file paths and line numbers.
Keep exploration minimal: stop reading as soon as you can answer."""

_GRAPH_INSTRUCTIONS = """\
You are a senior software engineer working inside a repository that has a
pre-built code knowledge graph.
The graph already indexed every symbol, call edge, import, and inheritance
relationship, so DO NOT crawl files. For almost any question, call
`explore` first: one call returns the matching symbols' verbatim source,
their callers/callees, and the impact radius. Use `repo_map` to survey the
codebase, `symbol_source` to read one symbol, and `impact_of` before
proposing edits. Only fall back to `read_file` for non-code files or when
the graph explicitly says a symbol is unknown.
Trust graph results; do not re-verify them by re-reading files.
Answer precisely, citing file paths and line numbers."""


def build_baseline_agent(
    root: Path, model: str | Model = DEFAULT_MODEL
) -> Agent[None]:
    """Agent limited to raw filesystem exploration."""
    root = root.resolve()

    @function_tool
    def list_dir(relative: str = ".") -> str:
        """List entries of a directory inside the repo (dirs end with /).

        Args:
            relative: Repo-relative directory path, e.g. "src/pkg".
        """
        return impl.list_dir(root, relative)

    @function_tool
    def read_file(relative: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a file slice with line numbers (max 400 lines per call).

        Args:
            relative: Repo-relative file path.
            start_line: First line to read (1-based).
            end_line: Last line to read; 0 means to end of file.
        """
        return impl.read_file(root, relative, start_line, end_line or None)

    @function_tool
    def grep(pattern: str, glob: str = "*.py") -> str:
        """Regex-search file contents; returns path:line: text matches.

        Args:
            pattern: Python regular expression.
            glob: Filename glob to restrict the search, default "*.py".
        """
        return impl.grep(root, pattern, glob)

    return Agent(
        name="baseline-coder",
        instructions=_BASELINE_INSTRUCTIONS,
        model=model,
        tools=[list_dir, read_file, grep],
    )


def build_graph_agent(
    root: Path,
    graph: CodeGraph,
    model: str | Model = DEFAULT_MODEL,
) -> Agent[None]:
    """Agent whose primary interface is the pre-built knowledge graph."""
    root = root.resolve()

    @function_tool
    def explore(query: str) -> str:
        """Answer almost any code question in one call.

        Returns the best-matching symbols' verbatim source, their
        callers/callees, subclasses, and impact radius. Use this first.

        Args:
            query: A symbol name, dotted name (Class.method), or keyword.
        """
        return impl.explore(graph, root, query)

    @function_tool
    def repo_map() -> str:
        """Compact overview of every file with its classes and functions."""
        return impl.repo_map(graph)

    @function_tool
    def symbol_source(symbol: str) -> str:
        """Verbatim line-numbered source of one symbol.

        Args:
            symbol: Symbol name or dotted name, e.g. "UserService.greet".
        """
        return impl.symbol_source(graph, root, symbol)

    @function_tool
    def impact_of(symbol: str, depth: int = 2) -> str:
        """Blast radius: everything that transitively depends on a symbol.

        Args:
            symbol: Symbol name to analyze.
            depth: Maximum reverse-dependency hops (default 2).
        """
        return impl.impact_of(graph, symbol, depth)

    @function_tool
    def read_file(relative: str, start_line: int = 1, end_line: int = 0) -> str:
        """Fallback reader for non-code files (max 400 lines per call).

        Args:
            relative: Repo-relative file path.
            start_line: First line to read (1-based).
            end_line: Last line to read; 0 means to end of file.
        """
        return impl.read_file(root, relative, start_line, end_line or None)

    return Agent(
        name="graph-coder",
        instructions=_GRAPH_INSTRUCTIONS,
        model=model,
        tools=[explore, repo_map, symbol_source, impact_of, read_file],
    )
