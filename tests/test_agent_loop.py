"""Agent loop tests using a scripted fake Client."""
 
from __future__ import annotations
 
import os
from dataclasses import dataclass, field
from typing import Any
 
import pytest
 
from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display
 
pytestmark = pytest.mark.asyncio
 
 
@dataclass
class FakeClient:
    script: list[CompletionResult]
    calls_seen: list[list[dict[str, Any]]] = field(default_factory=list)
    _i: int = 0
 
    async def health(self) -> bool:
        return True
 
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: Any = None,
    ) -> CompletionResult:
        self.calls_seen.append(list(messages))
        if self._i >= len(self.script):
            return CompletionResult(text="done.", tool_calls=[])
        result = self.script[self._i]
        self._i += 1
        return result
 
 
def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)
 
 
async def test_agent_writes_then_finishes(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 5
 
    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[_tc("write_file", {"path": "hi.py", "content": "print('hi')\n"})]
            ),
            CompletionResult(text="Wrote hi.py.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("create hi.py that prints 'hi'")
 
    assert result.success
    assert result.final_text == "Wrote hi.py."
    assert result.turns_used == 2
    assert "hi.py" in result.files_created
    assert os.path.isfile(tmp_path / "hi.py")
    # Two LLM calls: one produced the tool_call, one produced the final text
    assert len(fake.calls_seen) == 2
 
 
async def test_agent_stops_after_three_failures(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 10
 
    # Three attempts to read a non-existent file → 3 consecutive failures → stop
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("read_file", {"path": "nope1"}, call_id=f"c{i}")])
            for i in range(5)
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("read a bad file")
 
    assert not result.success
    assert "consecutive tool failures" in result.error
    # Should have stopped before all 5 scripted turns ran
    assert fake._i == 3
 
 
async def test_agent_refuses_write_in_plan_mode(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5
 
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("write_file", {"path": "x.py", "content": "x"})]),
            CompletionResult(text="ok stopping.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("write x.py")
 
    assert not (tmp_path / "x.py").exists()
    # Tool result message should communicate the refusal back to the model
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert any("plan mode" in (m.get("content") or "") for m in tool_msgs)
 
 


async def test_agent_runs_headless_without_display(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 5

    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[_tc("write_file", {"path": "a.py", "content": "x"})]
            ),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    result = await agent.run("write a.py")

    assert result.success
    assert (tmp_path / "a.py").read_text() == "x"


async def test_agent_allows_many_consecutive_reads(tmp_path):
    """Verify that many consecutive reads are allowed (no artificial limit)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 20

    # Create multiple files so reads succeed
    for i in range(15):
        (tmp_path / f"file{i}.txt").write_text(f"content {i}")

    script = [
        CompletionResult(tool_calls=[_tc("read_file", {"path": f"file{i}.txt"}, call_id=f"c{i}")])
        for i in range(15)
    ] + [CompletionResult(text="done.", tool_calls=[])]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("read many files")

    # All 15 reads should succeed followed by final text - no artificial refusal limit
    assert result.success
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    refusal_msgs = [m for m in tool_msgs if "refused" in (m.get("content") or "").lower()]
    assert not refusal_msgs, f"Should allow many reads without refusal: {refusal_msgs}"
