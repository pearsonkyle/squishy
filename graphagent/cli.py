"""Command-line interface.

Examples::

    graphagent index /path/to/repo            # build + save the graph
    graphagent map /path/to/repo              # print the repo map
    graphagent explore /path/to/repo slugify  # one-shot symbol exploration
    graphagent bench /path/to/repo "How does login work?" --runs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphagent import __version__
from graphagent.agentkit import tools
from graphagent.bench.harness import run_ab_sync
from graphagent.graph.builder import build_graph
from graphagent.graph.store import CodeGraph

_GRAPH_FILENAME = ".graphagent.json"


def _load_or_build(repo: Path) -> CodeGraph:
    cached = repo / _GRAPH_FILENAME
    if cached.is_file():
        return CodeGraph.load(cached)
    return build_graph(repo)


def _cmd_index(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    graph = build_graph(repo)
    graph.save(repo / _GRAPH_FILENAME)
    print(f"indexed {repo}: {graph.stats()}")
    return 0


def _cmd_map(args: argparse.Namespace) -> int:
    print(tools.repo_map(_load_or_build(Path(args.repo).resolve())))
    return 0


def _cmd_explore(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    print(tools.explore(_load_or_build(repo), repo, args.query))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    result = run_ab_sync(
        Path(args.repo).resolve(),
        args.task,
        runs_per_arm=args.runs,
        model=args.model,
        output=output,
        base_url=args.base_url,
    )
    print(json.dumps(result["comparison_of_medians"], indent=2))
    if output:
        print(f"full report written to {output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphagent")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build and cache the knowledge graph")
    p_index.add_argument("repo")
    p_index.set_defaults(func=_cmd_index)

    p_map = sub.add_parser("map", help="print the repo map")
    p_map.add_argument("repo")
    p_map.set_defaults(func=_cmd_map)

    p_explore = sub.add_parser("explore", help="one-shot symbol exploration")
    p_explore.add_argument("repo")
    p_explore.add_argument("query")
    p_explore.set_defaults(func=_cmd_explore)

    p_bench = sub.add_parser("bench", help="run the A/B comparison")
    p_bench.add_argument("repo")
    p_bench.add_argument("task")
    p_bench.add_argument("--runs", type=int, default=1, help="runs per arm")
    p_bench.add_argument("--model", default=None, help="override model name")
    p_bench.add_argument("--output", default=None, help="write full JSON report")
    p_bench.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://localhost:1234/v1. "
             "--model is then required: the endpoint loads whatever id it is given.",
    )
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        # Output was piped to a consumer (head, less) that closed early.
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
