"""Agent loop tests using a scripted fake Client."""
 
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display
from squishy.plan_state import plan_path
 
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

    # Even if the model hallucinates a write_file call (it isn't in the plan-mode
    # schemas), dispatch-level defence still blocks the mutation.
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("write_file", {"path": "x.py", "content": "x"})]),
            CompletionResult(
                tool_calls=[
                    _tc(
                        "plan_task",
                        {
                            "problem": "user asked for x.py",
                            "solution": "create it",
                            "steps": ["write x.py"],
                        },
                        call_id="c2",
                    )
                ]
            ),
            CompletionResult(text="ok stopping.", tool_calls=[]),
        ]
    )

    async def auto_approve(_tool, _args):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    await agent.run("write x.py")

    assert not (tmp_path / "x.py").exists()


async def test_agent_plan_mode_schemas_exclude_writes(tmp_path):
    """The LLM in plan mode should not see write_file/edit_file in tool schemas."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 3

    # Capture the tools list passed to complete()
    captured_tools: list[list[dict]] = []

    @dataclass
    class CapturingClient:
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
            captured_tools.append(list(tools))
            self._i += 1
            return CompletionResult(
                tool_calls=[
                    _tc(
                        "plan_task",
                        {
                            "problem": "p",
                            "solution": "s",
                            "steps": ["a", "b"],
                        },
                    )
                ]
            ) if self._i == 1 else CompletionResult(text="done", tool_calls=[])

    async def auto_approve(_tool, _args):
        return True

    agent = Agent(cfg, CapturingClient(), Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    await agent.run("plan something")

    assert captured_tools, "complete() was never called"
    names = {t["function"]["name"] for t in captured_tools[0]}
    assert "plan_task" in names
    assert "write_file" not in names
    assert "edit_file" not in names


async def test_agent_plan_mode_requires_plan_task(tmp_path):
    """If plan mode finishes with prose and no plan_task, the agent should nudge."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 10

    fake = FakeClient(
        script=[
            CompletionResult(text="I think the fix is obvious.", tool_calls=[]),
            CompletionResult(text="Still no plan.", tool_calls=[]),
            CompletionResult(
                tool_calls=[
                    _tc(
                        "plan_task",
                        {"problem": "p", "solution": "s", "steps": ["a"]},
                    )
                ]
            ),
            CompletionResult(text="shouldn't reach", tool_calls=[]),
        ]
    )

    async def auto_approve(_tool, _args):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    result = await agent.run("do the thing")

    assert result.success, result.error
    # Agent must have produced a plan before finishing
    plan = agent.tool_ctx.plan
    assert plan is not None
    assert plan.approved is True
    # Nudge messages should appear in the transcript
    nudge_msgs = [
        m
        for m in result.messages
        if m.get("role") == "user" and "[system]" in (m.get("content") or "")
    ]
    assert nudge_msgs, "expected at least one nudge injection"
    assert any("partial or empty if uncertain" in (m.get("content") or "") for m in nudge_msgs)


async def test_agent_plan_mode_gives_up_after_nudges(tmp_path):
    """If the model keeps refusing to plan, the agent should eventually stop."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 10

    fake = FakeClient(
        script=[CompletionResult(text=f"prose {i}", tool_calls=[]) for i in range(6)]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("plan please")

    assert not result.success
    assert "plan_task" in result.error


async def test_agent_completes_when_plan_task_approved(tmp_path):
    """A successful plan_task + user approval should terminate the run."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5

    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[
                    _tc(
                        "plan_task",
                        {
                            "plan": "Fix the bug",
                            "problem": "Bug in foo",
                            "solution": "Patch it",
                            "steps": ["read foo", "edit foo"],
                        },
                    )
                ]
            ),
            CompletionResult(text="extra turn that should not run", tool_calls=[]),
        ]
    )

    async def auto_approve(_tool, _args):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    result = await agent.run("please plan")

    assert result.success
    assert "Plan approved" in result.final_text
    plan = agent.tool_ctx.plan
    assert plan is not None
    assert plan.approved is True
    # The second scripted completion should not have been consumed
    assert fake._i == 1
 
 


async def test_agent_plan_mode_nudges_after_tool_turns(tmp_path):
    """In plan mode, reading files for MAX_PLAN_TOOL_TURNS turns without calling
    plan_task should inject a nudge, then eventually produce the plan."""
    from squishy.agent import MAX_PLAN_TOOL_TURNS

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 30

    # Create a file so read_file succeeds
    (tmp_path / "foo.py").write_text("# code")

    # MAX_PLAN_TOOL_TURNS turns of read-only tool calls, then plan_task
    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id=f"c{i}")]
        )
        for i in range(MAX_PLAN_TOOL_TURNS)
    ] + [
        CompletionResult(
            tool_calls=[
                _tc(
                    "plan_task",
                    {"problem": "p", "solution": "s", "steps": ["a"]},
                    call_id="plan1",
                )
            ]
        ),
        CompletionResult(text="should not reach", tool_calls=[]),
    ]

    async def auto_approve(_tool, _args):
        return True

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    result = await agent.run("plan something")

    assert result.success, result.error
    plan = agent.tool_ctx.plan
    assert plan is not None
    assert plan.approved is True
    # A nudge message should have been injected
    nudge_msgs = [
        m
        for m in result.messages
        if m.get("role") == "user" and "[system]" in (m.get("content") or "")
        and "read tools" in (m.get("content") or "")
    ]
    assert nudge_msgs, "expected at least one tool-turn nudge injection"
    assert any("partial or empty if uncertain" in (m.get("content") or "") for m in nudge_msgs)


async def test_agent_plan_mode_gives_up_after_tool_turn_nudges(tmp_path):
    """If the model keeps calling read tools without ever calling plan_task, the
    agent should give up after exhausting nudge budget (tool-call path)."""
    from squishy.agent import MAX_PLAN_NUDGES, MAX_PLAN_TOOL_TURNS

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 30

    (tmp_path / "foo.py").write_text("# code")

    # Enough turns to exhaust all nudges: (MAX_PLAN_NUDGES + 1) * MAX_PLAN_TOOL_TURNS
    n_turns = (MAX_PLAN_NUDGES + 1) * MAX_PLAN_TOOL_TURNS + 1
    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id=f"c{i}")]
        )
        for i in range(n_turns)
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("plan please")

    assert not result.success
    assert "plan_task" in result.error


async def test_agent_restores_persisted_plan_state(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5

    fake = FakeClient(
        script=[
            CompletionResult(
                tool_calls=[
                    _tc(
                        "plan_task",
                        {"problem": "p", "solution": "s", "steps": ["read foo", "edit foo"]},
                    )
                ]
            )
        ]
    )

    async def auto_approve(_tool, _args):
        return True

    await Agent(cfg, fake, Display(), prompt_fn=auto_approve).run("plan it")  # type: ignore[arg-type]
    assert plan_path(tmp_path).is_file()

    restored = Agent(cfg, FakeClient(script=[]), Display())  # type: ignore[arg-type]
    assert restored.tool_ctx.plan is not None
    assert restored.tool_ctx.plan.approved is True
    assert restored.tool_ctx.plan.steps[0].description == "read foo"


async def test_agent_blocks_success_until_plan_steps_resolved(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 6

    plan_json = {
        "id": "plan-test",
        "plan": "Fix bug",
        "problem": "p",
        "solution": "s",
        "approved": True,
        "steps": [
            {"id": "step-1", "description": "edit file", "status": "pending"},
        ],
    }
    plan_path(tmp_path).parent.mkdir(exist_ok=True)
    plan_path(tmp_path).write_text(json.dumps(plan_json))

    fake = FakeClient(
        script=[
            CompletionResult(text="done", tool_calls=[]),
            CompletionResult(tool_calls=[_tc("update_plan", {"step_index": 1, "status": "done"})]),
            CompletionResult(text="really done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("finish the task")

    assert result.success
    assert result.final_text == "really done"
    assert result.plan_state is not None
    assert result.plan_state["progress"]["done"] == 1
    persisted = json.loads(plan_path(tmp_path).read_text())
    assert persisted["steps"][0]["status"] == "done"
    assert any("unresolved steps" in (m.get("content") or "") for m in result.messages)


async def test_agent_plan_mode_without_index_does_not_force_recall(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5
    (tmp_path / "foo.py").write_text("# code")

    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("read_file", {"path": "foo.py"})]),
            CompletionResult(tool_calls=[_tc("plan_task", {"problem": "p", "solution": "s", "steps": ["a"]})]),
        ]
    )

    async def auto_approve(_tool, _args):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    result = await agent.run("plan without index")

    assert result.success
    assert not any("Too many read calls without `recall`" in (m.get("content") or "") for m in result.messages)


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
