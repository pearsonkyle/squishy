"""On AgentTimeout, the in-progress transcript and turn_log must be
preserved on the exception so the bench harness can recover them.

Before this fix, AgentTimeout was raised with no payload — the bench
harness's exception handler set ``task_result = None`` and the prediction
JSON ended up with an empty ``transcript`` field, hiding everything the
agent did up to the timeout.  This was exactly what blocked diagnosis of
v27p3 scico-561 (which timed out at 900s with no recoverable trace).
"""
from __future__ import annotations

import asyncio

from conftest import FakeClient

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.errors import AgentTimeout


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


class _SlowClient(FakeClient):
    """FakeClient that sleeps before returning the Nth call (1-indexed)."""

    def __init__(self, script, slow_after: int = 1, sleep_s: float = 10.0):
        super().__init__(script=script)
        self._slow_after = slow_after
        self._sleep_s = sleep_s
        self._n = 0

    async def complete(self, *args, **kwargs):  # type: ignore[override]
        self._n += 1
        if self._n > self._slow_after:
            await asyncio.sleep(self._sleep_s)
        return await super().complete(*args, **kwargs)


async def test_agent_timeout_carries_partial_result(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 10

    # First completion writes a file (returns immediately).  Second call
    # sleeps long enough to trigger the wall-clock timeout.
    script = [
        CompletionResult(tool_calls=[
            _tc("write_file", {"path": "hi.py", "content": "x = 1\n"}),
        ]),
        CompletionResult(text="should not reach", tool_calls=[]),
    ]
    fake = _SlowClient(script=script, slow_after=1, sleep_s=10.0)
    agent = Agent(cfg, fake)  # type: ignore[arg-type]

    raised: AgentTimeout | None = None
    try:
        await agent.run("write hi.py", timeout=1.0)
    except AgentTimeout as e:
        raised = e

    assert raised is not None, "expected AgentTimeout"
    partial = getattr(raised, "partial_result", None)
    assert partial is not None, (
        "AgentTimeout must carry a partial_result so the bench harness "
        "can recover the transcript built up to the timeout"
    )
    assert partial.success is False
    # The first turn's write_file landed before the slow second call —
    # it must show up in the recovered partial result.
    assert "hi.py" in partial.files_created, (
        f"partial result lost write_file evidence: {partial.files_created!r}"
    )
    # full_log should contain at least system + user + assistant + tool
    # result.  Empty would mean diagnostics are still being discarded.
    assert len(partial.full_log) >= 4, (
        f"partial.full_log too small: {len(partial.full_log)}"
    )
    assert partial.error.startswith("AgentTimeout"), partial.error
