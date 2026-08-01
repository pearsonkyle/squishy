#!/usr/bin/env python3
"""In-container driver for the OpenAI-Agents-SDK arms.

Drop-in alternative to ``driver.py``: same task file, same result file, same
metric names, so ``run_bench.py`` grades both harnesses through one code path
and the numbers are directly comparable. The arm is chosen by the task's
``tool_profile``:

* ``sdk``   — filesystem discovery (grep/list_dir/read_file) plus edit tools.
* ``graph`` — the same agent with the pre-built knowledge graph in front of
  the crawl (``explore``/``impact_of``/``repo_map``).

Graph build time is recorded separately from the run: it is a one-off
preprocessing cost that a real deployment amortizes across every task in the
repo, and folding it into the agent's wall clock would flatter the baseline.

Metrics are flushed after every tool call. The paths that matter most — the
turn cap, the wall-clock kill, an upstream API error — are exactly the ones
where reading them off a returned result gets you nothing.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks, Runner
from agents.tool import Tool

from graphagent.agentkit.llm import resolve_model
from graphagent.agentkit.metrics import ToolTrace
from graphagent.agentkit.swe import build_swe_agent
from graphagent.graph.builder import build_graph

# Overridable so the driver can be smoke-tested outside a container, which is
# the difference between finding an import error in two seconds and finding it
# after a multi-gigabyte image pull.
TASK = Path(os.environ.get("SQUISHY_TASK_FILE", "/opt/squishy-task.json"))
OUT = Path(os.environ.get("SQUISHY_RESULT_FILE", "/opt/squishy-result.json"))


def _bucket(summary: str) -> str:
    """Collapse a failed call to a short cause so counts aggregate."""
    s = summary.lower()
    for marker, label in (
        ("not found", "old_str not found"),
        ("appears", "old_str not unique"),
        ("no such file", "path not found"),
        ("timed out", "timeout"),
        ("escapes", "path escapes root"),
    ):
        if marker in s:
            return label
    return "failed"


class Recorder(RunHooks[Any]):
    """Counts LLM turns and flushes partial metrics after every tool call."""

    def __init__(self, trace: ToolTrace, state: dict) -> None:
        super().__init__()
        self.trace = trace
        self.state = state

    async def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.state["turns"] += 1
        _flush(self.state, self.trace)

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
        result: str,
    ) -> None:
        _flush(self.state, self.trace)


def _metrics(state: dict, trace: ToolTrace) -> dict:
    events = trace.events
    failures: dict[str, int] = {}
    for event in events:
        if not event.ok:
            key = f"{event.name}: {_bucket(event.error)}"
            failures[key] = failures.get(key, 0) + 1
    return {
        "exit_status": state["exit_status"],
        "turns": state["turns"],
        "prompt_tokens": state["prompt_tokens"],
        "completion_tokens": state["completion_tokens"],
        "tool_calls": trace.total,
        "tool_counts": trace.by_tool,
        "tool_failures": trace.failures,
        "failure_reasons": failures,
        "trace": [f"{e.name}{'' if e.ok else '!'}" for e in events],
        "commands": trace.commands[-120:],
        "graph_build_s": state["graph_build_s"],
        "nudges": state["nudges"],
        "graph_stats": state["graph_stats"],
        "arm": state["arm"],
        "turn_budget": _effective_budget(state)[1],
    }


def _flush(state: dict, trace: ToolTrace) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(_metrics(state, trace)))
    tmp.replace(OUT)


def _dirty(repo: str) -> bool:
    p = subprocess.run(
        ["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True
    )
    return any(
        line for line in p.stdout.splitlines()
        if line.strip() and ".squishy" not in line and ".graphagent" not in line
    )


def _effective_budget(state: dict) -> tuple[int, int]:
    """(turns used, turns actually available) — the *binding* limit of the two.

    ``max_turns`` and ``task_timeout`` are set independently, and on a slow
    image they disagree: qiskit-terra-5662 runs at ~30s a turn under
    emulation, so a 50-turn budget under a 1200s wall is really a 40-turn
    budget. The agent was being told it had nine turns left at the moment the
    process was killed — the harness promising something it then takes away,
    which is the failure mode this whole harness keeps rediscovering.

    Projected from the observed per-turn cost, so it tracks the machine it is
    actually running on rather than a guess.
    """
    used = state["turns"]
    nominal = int(state["max_turns"])
    budget = nominal
    timeout = state["task_timeout"]
    if timeout and used > 0:
        per_turn = (time.time() - state["started_at"]) / used
        if per_turn > 0:
            # The `used + 1` floor keeps the notice from reading "turn 40 of
            # 38" once a run overshoots the projection, but it is clamped to
            # `nominal`: without that, the last turn of a 50-turn run reported
            # "50 of 51" and promised a turn that does not exist — the same
            # bug, one turn wide.
            projected = int(float(timeout) / per_turn)
            budget = min(nominal, max(used + 1, min(nominal, projected)))
    return used, budget


_EMPTY_RESPONSE_NUDGE = (
    "You returned an empty message and the working tree is unchanged, so "
    "nothing has been fixed yet. Keep going: make the source edit."
)


async def _drive(
    agent: Agent[None],
    task: dict,
    hooks: Recorder,
    state: dict,
    repo: str,
) -> str:
    """Run the agent until the tree changes, the budget runs out, or it quits.

    ``Runner.run`` returns as soon as the model emits a message with no tool
    call — including an *empty* message, which this model does regularly. The
    SDK treats that as a finished task; on cfn-lint-3965 both arms ended that
    way, with a scratch file in /tmp, no edit, and an empty final answer. So
    an unchanged tree is not accepted as an ending: the conversation is
    resumed, in place, with a nudge appended.

    Resumed rather than restarted. ``result.to_input_list()`` carries the
    whole transcript forward, so the model keeps everything it learned; the
    alternative — a fresh ``Runner.run`` with a nudge as the first message —
    throws away the exploration that got it that far.

    The only bound on continuations is the turn budget. A count-based cap was
    tried first and is worse than it sounds: each of this model's segments
    ends after a handful of turns, so on qiskit-terra-5662 three nudges were
    spent by turn 21 of 50 and the run was declared over with 29 turns of
    budget unused and no patch. ``empty_patch_retries=0`` still disables
    continuation entirely, which is how you measure the raw SDK behavior.
    """
    budget = int(task["max_turns"])
    timeout = task.get("task_timeout")
    nudging = int(task.get("empty_patch_retries", 0)) > 0
    items: Any = task["prompt"]
    final_text = ""

    while True:
        remaining = budget - state["turns"]
        if remaining <= 0:
            state["exit_status"] = "max_turns"
            return final_text
        before = state["turns"]
        result = await asyncio.wait_for(
            Runner.run(agent, items, hooks=hooks, max_turns=remaining), timeout
        )
        usage = result.context_wrapper.usage
        state["prompt_tokens"] += usage.input_tokens
        state["completion_tokens"] += usage.output_tokens
        final_text = str(result.final_output or "") or final_text
        state["exit_status"] = "completed"

        if _dirty(repo) or not nudging:
            return final_text
        if state["turns"] == before:
            # A segment that burned no turns cannot make progress, and with a
            # budget-bounded loop it would spin forever.
            state["exit_status"] = "stalled"
            return final_text
        if budget - state["turns"] <= 0:
            # Checked here as well as at the top so `nudges` counts nudges the
            # model actually saw, not one it had no budget to answer.
            state["exit_status"] = "max_turns"
            return final_text
        state["nudges"] += 1
        nudge = (
            _EMPTY_RESPONSE_NUDGE
            if not str(result.final_output or "").strip()
            else task["empty_patch_nudge"]
        )
        items = result.to_input_list() + [{"role": "user", "content": nudge}]


async def main() -> int:
    task = json.loads(TASK.read_text())
    repo = task["repo"]
    arm = task.get("tool_profile", "graph")

    trace = ToolTrace()
    state = {
        "exit_status": "running", "turns": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "graph_build_s": 0.0, "graph_stats": {},
        "arm": arm, "nudges": 0, "started_at": time.time(),
        "max_turns": int(task["max_turns"]),
        "task_timeout": task.get("task_timeout"),
    }
    _flush(state, trace)

    graph = None
    if arm == "graph":
        started = time.perf_counter()
        graph = build_graph(Path(repo))
        state["graph_build_s"] = round(time.perf_counter() - started, 1)
        state["graph_stats"] = graph.stats()
        _flush(state, trace)

    agent = build_swe_agent(
        Path(repo),
        graph=graph,
        model=resolve_model(task["model"], base_url=task["base_url"]),
        trace=trace,
        # Lets a tool result say "turn 38 of 50 and nothing edited yet". The
        # SDK never tells the model the clock is running.
        progress=lambda: _effective_budget(state),
    )
    hooks = Recorder(trace, state)

    t0 = time.time()
    final_text = ""
    error = ""
    try:
        final_text = await _drive(agent, task, hooks, state, repo)
    except TimeoutError:
        state["exit_status"] = "wall_timeout"
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        state["exit_status"] = (
            "max_turns" if "MaxTurns" in name else f"error:{name}"
        )
        error = f"{name}: {exc}"

    payload = _metrics(state, trace)
    payload.update({
        "final_text": final_text[-4000:],
        "error": error,
        "agent_elapsed_s": round(time.time() - t0, 1),
    })
    OUT.write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
