"""Tests for run metrics, agent construction, and A/B comparison math."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from graphagent import __version__
from graphagent.agentkit.factory import build_baseline_agent, build_graph_agent
from graphagent.agentkit.metrics import RunMetrics, ToolCallRecorder
from graphagent.bench.harness import compare
from graphagent.graph.builder import build_graph


def test_version_is_semver() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_tool_call_recorder_counts_thread_safely() -> None:
    recorder = ToolCallRecorder()

    def hammer() -> None:
        for _ in range(500):
            recorder.record("grep")

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert recorder.total == 8 * 500
    assert recorder.by_tool["grep"] == 8 * 500


def test_recorder_hook_integrates_with_sdk_signature() -> None:
    """The SDK calls ``on_tool_start(ctx, agent, tool)``; verify our hook."""

    class _Tool:
        name = "read_file"

    recorder = ToolCallRecorder()
    asyncio.run(recorder.on_tool_start(None, None, _Tool()))  # type: ignore[arg-type]
    assert recorder.by_tool == {"read_file": 1}


def test_baseline_agent_has_filesystem_tools(sample_repo: Path) -> None:
    agent = build_baseline_agent(sample_repo)
    names = {t.name for t in agent.tools}
    assert names == {"list_dir", "read_file", "grep"}


def test_graph_agent_has_graph_and_edit_tools(sample_repo: Path) -> None:
    graph = build_graph(sample_repo)
    agent = build_graph_agent(sample_repo, graph)
    names = {t.name for t in agent.tools}
    assert {"explore", "repo_map", "symbol_source", "impact_of", "read_file"} <= names


def test_compare_computes_deltas() -> None:
    baseline = RunMetrics(
        arm="baseline",
        task="t",
        tool_calls=20,
        tool_calls_by_name={"grep": 10, "read_file": 10},
        llm_requests=8,
        input_tokens=90_000,
        output_tokens=4_000,
        wall_seconds=30.0,
        final_output="a",
    )
    graph = RunMetrics(
        arm="graph",
        task="t",
        tool_calls=5,
        tool_calls_by_name={"explore": 3, "read_file": 2},
        llm_requests=4,
        input_tokens=30_000,
        output_tokens=3_000,
        wall_seconds=15.0,
        final_output="b",
    )
    report = compare(baseline, graph)
    assert report["tool_calls"]["delta_pct"] == -75.0
    assert report["total_tokens"]["baseline"] == 94_000
    assert report["total_tokens"]["graph"] == 33_000
    assert report["wall_seconds"]["delta_pct"] == -50.0
