"""A/B harness: run the same task through both arms and compare.

Methodology follows CodeGraph's published benchmark shape: same question
per repo, N runs per arm, median reported, comparing tool calls, LLM
requests, tokens, and wall time.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from agents import Agent, Runner

from graphagent.agentkit.factory import build_baseline_agent, build_graph_agent
from graphagent.agentkit.llm import resolve_model
from graphagent.agentkit.metrics import RunMetrics, ToolCallRecorder
from graphagent.graph.builder import build_graph
from graphagent.graph.store import CodeGraph

_MAX_TURNS = 25


async def run_once(agent: Agent[None], arm: str, task: str) -> RunMetrics:
    """Execute one agent run and capture its metrics."""
    recorder = ToolCallRecorder()
    started = time.perf_counter()
    error: str | None = None
    final_output = ""
    requests = input_tokens = output_tokens = 0
    try:
        result = await Runner.run(agent, task, hooks=recorder, max_turns=_MAX_TURNS)
        final_output = str(result.final_output)
        usage = result.context_wrapper.usage
        requests = usage.requests
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
    except Exception as exc:  # pragma: no cover - surfaced in results JSON
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    return RunMetrics(
        arm=arm,
        task=task,
        tool_calls=recorder.total,
        tool_calls_by_name=recorder.by_tool,
        llm_requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=round(elapsed, 2),
        final_output=final_output,
        error=error,
    )


def _delta(baseline: float, graph: float) -> dict[str, float]:
    pct = 0.0 if baseline == 0 else round((graph - baseline) / baseline * 100, 1)
    return {"baseline": baseline, "graph": graph, "delta_pct": pct}


def compare(baseline: RunMetrics, graph: RunMetrics) -> dict[str, Any]:
    """Head-to-head report between one baseline run and one graph run."""
    return {
        "task": baseline.task,
        "tool_calls": _delta(baseline.tool_calls, graph.tool_calls),
        "llm_requests": _delta(baseline.llm_requests, graph.llm_requests),
        "input_tokens": _delta(baseline.input_tokens, graph.input_tokens),
        "output_tokens": _delta(baseline.output_tokens, graph.output_tokens),
        "total_tokens": _delta(baseline.total_tokens, graph.total_tokens),
        "wall_seconds": _delta(baseline.wall_seconds, graph.wall_seconds),
    }


def _median_metrics(runs: list[RunMetrics]) -> RunMetrics:
    """Synthetic run holding the per-field median across ``runs``."""
    ok = [r for r in runs if r.error is None] or runs
    return RunMetrics(
        arm=ok[0].arm,
        task=ok[0].task,
        tool_calls=int(statistics.median(r.tool_calls for r in ok)),
        tool_calls_by_name={},
        llm_requests=int(statistics.median(r.llm_requests for r in ok)),
        input_tokens=int(statistics.median(r.input_tokens for r in ok)),
        output_tokens=int(statistics.median(r.output_tokens for r in ok)),
        wall_seconds=round(statistics.median(r.wall_seconds for r in ok), 2),
        final_output=ok[0].final_output,
    )


async def run_ab(
    repo: Path,
    task: str,
    runs_per_arm: int = 1,
    model: str | None = None,
    graph: CodeGraph | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Run the full A/B: build the graph, run both arms, report medians.

    Graph build time is reported separately — it is a one-time,
    amortized preprocessing cost, not part of a run.
    """
    repo = repo.resolve()
    build_started = time.perf_counter()
    graph = graph or build_graph(repo)
    build_seconds = round(time.perf_counter() - build_started, 2)

    resolved = resolve_model(model, base_url=base_url)
    baseline_agent = build_baseline_agent(repo, model=resolved)
    graph_agent = build_graph_agent(repo, graph, model=resolved)

    baseline_runs = [
        await run_once(baseline_agent, "baseline", task) for _ in range(runs_per_arm)
    ]
    graph_runs = [
        await run_once(graph_agent, "graph", task) for _ in range(runs_per_arm)
    ]

    report = compare(_median_metrics(baseline_runs), _median_metrics(graph_runs))
    return {
        "repo": str(repo),
        "model": model or "default",
        "graph_stats": graph.stats(),
        "graph_build_seconds": build_seconds,
        "runs_per_arm": runs_per_arm,
        "comparison_of_medians": report,
        "baseline_runs": [r.to_dict() for r in baseline_runs],
        "graph_runs": [r.to_dict() for r in graph_runs],
    }


def run_ab_sync(
    repo: Path,
    task: str,
    runs_per_arm: int = 1,
    model: str | None = None,
    output: Path | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Synchronous entry point; optionally writes the report to JSON."""
    result = asyncio.run(
        run_ab(repo, task, runs_per_arm=runs_per_arm, model=model, base_url=base_url)
    )
    if output is not None:
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    return result
