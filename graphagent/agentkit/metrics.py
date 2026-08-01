"""Instrumentation for A/B runs: tool-call counts and token usage."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.tool import Tool


class ToolCallRecorder(RunHooks[Any]):
    """SDK run hook that counts every tool invocation, thread-safely.

    The Agents SDK may execute tools concurrently, so the counters are
    guarded by a lock.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._by_tool: dict[str, int] = {}

    def record(self, tool_name: str) -> None:
        with self._lock:
            self._by_tool[tool_name] = self._by_tool.get(tool_name, 0) + 1

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
    ) -> None:
        self.record(tool.name)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._by_tool.values())

    @property
    def by_tool(self) -> dict[str, int]:
        with self._lock:
            return dict(self._by_tool)


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """One tool call: what it was asked to do, and whether that worked."""

    name: str
    summary: str
    ok: bool = True
    error: str = ""


class ToolTrace:
    """Argument-level record of every tool call, thread-safely.

    ``RunHooks.on_tool_start`` gets the tool but not its arguments, and the
    arguments *are* the trajectory: a run of forty ``read_file`` calls says
    nothing about why it produced no patch, while the paths and the failing
    ``old_str`` say all of it. So the factory's tool wrappers report here
    directly, which also lets a tool record its own failure — the SDK sees a
    string return either way.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[ToolEvent] = []

    def record(
        self, name: str, summary: str, ok: bool = True, error: str = ""
    ) -> None:
        with self._lock:
            self._events.append(ToolEvent(name, summary, ok, error))

    @property
    def events(self) -> list[ToolEvent]:
        with self._lock:
            return list(self._events)

    @property
    def total(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def failures(self) -> int:
        with self._lock:
            return sum(1 for e in self._events if not e.ok)

    @property
    def by_tool(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.name] = counts.get(event.name, 0) + 1
        return counts

    @property
    def commands(self) -> list[str]:
        """One dense line per call, in order — the readable trajectory."""
        return [
            f"{i}{'' if e.ok else '!'}: {e.name} {e.summary}"
            + (f"  <- {e.error}" if e.error else "")
            for i, e in enumerate(self.events, start=1)
        ]


@dataclass(slots=True)
class RunMetrics:
    """Everything we compare between the two arms for one run."""

    arm: str
    task: str
    tool_calls: int
    tool_calls_by_name: dict[str, int]
    llm_requests: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    final_output: str
    error: str | None = field(default=None)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task": self.task,
            "tool_calls": self.tool_calls,
            "tool_calls_by_name": self.tool_calls_by_name,
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "wall_seconds": self.wall_seconds,
            "final_output": self.final_output,
            "error": self.error,
        }
