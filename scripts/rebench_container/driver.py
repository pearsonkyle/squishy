#!/usr/bin/env python3
"""In-container agent driver.

Runs inside the instance's own image, in the repo checkout, with the real
toolchain on PATH — so `run_command` behaves the way it does for a real user.
Talks to the LLM on the host over `host.docker.internal`.

Metrics are accumulated from the agent's event stream as it runs, and the
result file is rewritten after every event. Sourcing them from the returned
``TaskResult`` alone loses everything on the paths weak models hit most: the
turn cap, the wall-clock timeout, and upstream API errors. If this process is
killed mid-run, the partial metrics on disk are still correct.

Invoked by run_bench.py; not meant to be run by hand.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from squishy.api import Squishy

TASK = Path("/opt/squishy-task.json")
OUT = Path("/opt/squishy-result.json")


def _bucket(err: str) -> str:
    """Collapse a tool error to a short cause, so counts aggregate.

    Errors embed paths, line numbers, and snippets of the model's own text;
    without this every failure is unique and the tally says nothing.
    """
    e = err.lower()
    for marker, label in (
        ("not found in", "old_string not found"),
        ("appears", "old_string not unique"),
        ("already exists", "file exists"),
        ("no such file", "missing path"),
        ("refused:", "refused"),
        ("read the file", "read before edit"),
        ("identical", "repeated identical call"),
        ("timed out", "timeout"),
        ("unknown tool", "unknown tool"),
    ):
        if marker in e:
            return label
    return (err.split("\n")[0][:60] or "unknown")


class Metrics:
    """Event-stream accumulator (the RunHooks equivalent)."""

    def __init__(self) -> None:
        self.turns = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = 0
        self.tool_counts: dict[str, int] = {}
        self.tool_failures = 0
        self.failure_reasons: dict[str, int] = {}
        self.commands: list[str] = []
        # (turn, tool) so we can see *when* a tool ran, not just how often.
        self.trace: list[str] = []
        self._turn = 0
        # Each sq.run() restarts turn numbering at 1. Without an offset the
        # empty-patch retry's turns vanish from the totals — exactly the
        # "metrics from the final result" failure, one level up.
        self._base = 0
        self.exit_status = "running"

    def on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "turn":
            if ev["turn"] == 1 and self._turn:
                self._base = self.turns
            self._turn = self._base + ev["turn"]
            self.turns = self._turn
        elif kind == "usage":
            # Per-turn deltas, not the run's running total — the total also
            # restarts on the retry run.
            self.prompt_tokens += ev["prompt_tokens"]
            self.completion_tokens += ev["completion_tokens"]
        elif kind == "tool":
            name = ev["name"]
            ok = bool(ev.get("success"))
            self.tool_calls += 1
            self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
            if not ok:
                self.tool_failures += 1
                key = f"{name}: {_bucket(ev.get('error') or '')}"
                self.failure_reasons[key] = self.failure_reasons.get(key, 0) + 1
            self.trace.append(f"{self._turn}:{name}" + ("" if ok else "!"))
            # With a shell-only profile the command *is* the trajectory —
            # a trace of 76 identical "run_command" entries says nothing about
            # why a run produced no patch.
            if name == "run_command":
                cmd = " ".join(str((ev.get("args") or {}).get("command", "")).split())
                self.commands.append(f"{self._turn}{'' if ok else '!'}: {cmd[:160]}")
        self.flush()

    def as_dict(self) -> dict:
        return {
            "exit_status": self.exit_status,
            "turns": self.turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
            "tool_counts": self.tool_counts,
            "tool_failures": self.tool_failures,
            "failure_reasons": self.failure_reasons,
            "trace": self.trace,
            "commands": self.commands[-120:],
        }

    def flush(self) -> None:
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.as_dict()))
        tmp.replace(OUT)


async def main() -> int:
    task = json.loads(TASK.read_text())
    m = Metrics()
    m.flush()

    t0 = time.time()
    final_text = ""
    error = ""
    try:
        async with Squishy(
            model=task["model"],
            base_url=task["base_url"],
            api_key=task.get("api_key", "local"),
            permission_mode=task.get("mode", "bench"),
            tool_profile=task.get("tool_profile", "standard"),
            max_turns=task["max_turns"],
            request_timeout=task.get("request_timeout", 180.0),
            max_retries=task.get("max_retries", 3),
            max_turns_without_edit=task.get("max_turns_without_edit", 12),
            # The agent is already inside the instance container; nesting a
            # Docker sandbox here would be both impossible and pointless.
            use_sandbox=False,
            save_sessions=False,
        ) as sq:
            res = await sq.run(
                task["prompt"],
                working_dir=task["repo"],
                timeout=task.get("task_timeout"),
                on_event=m.on_event,
                notes={
                    "fail_to_pass_tests": json.dumps(task.get("fail_to_pass") or []),
                    "test_cmd": task.get("test_cmd", ""),
                },
            )
            final_text = res.final_text
            m.exit_status = "completed" if res.success else "incomplete"
            if res.error:
                error = res.error
                m.exit_status = "max_turns" if "max turns" in res.error.lower() else "error"

            # Empty-patch retry, bounded. Fires only on a clean exit with an
            # unchanged tree: "explored, understood the bug, quit without
            # editing" is the dominant small-model failure, and one shove
            # usually clears it.
            for _ in range(task.get("empty_patch_retries", 0)):
                if _dirty(task["repo"]):
                    break
                res = await sq.run(
                    task["empty_patch_nudge"], working_dir=task["repo"],
                    timeout=task.get("task_timeout"), on_event=m.on_event,
                )
                final_text = res.final_text or final_text
    except TimeoutError:
        m.exit_status = "wall_timeout"
    except Exception as e:  # noqa: BLE001
        m.exit_status = f"error:{type(e).__name__}"
        error = f"{type(e).__name__}: {e}"

    payload = m.as_dict()
    payload.update({
        "final_text": final_text[-4000:],
        "error": error,
        "agent_elapsed_s": round(time.time() - t0, 1),
    })
    OUT.write_text(json.dumps(payload))
    return 0


def _dirty(repo: str) -> bool:
    import subprocess
    p = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                       capture_output=True, text=True)
    return any(
        line for line in p.stdout.splitlines()
        if line.strip() and ".squishy" not in line
    )


if __name__ == "__main__":
    os.environ.setdefault("SQUISHY_SAVE_SESSIONS", "0")
    sys.exit(asyncio.run(main()))
