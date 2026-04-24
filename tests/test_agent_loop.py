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
    cfg.max_turns = 20
    cfg.max_consecutive_errors = 3  # override default for this test

    # Three attempts to read non-existent files → 3 consecutive failures → stop.
    # Use different paths each time so the quality monitor doesn't flag them
    # as repeated identical calls.
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("read_file", {"path": f"nope{i}"}, call_id=f"c{i}")])
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
    """In plan mode, reading files for max_plan_investigation_turns turns without
    calling plan_task should inject a nudge, then eventually produce the plan."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 30
    max_tool_turns = cfg.max_plan_investigation_turns

    # Create a file so read_file succeeds
    (tmp_path / "foo.py").write_text("# code")

    # max_plan_investigation_turns turns of read-only tool calls, then plan_task
    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id=f"c{i}")]
        )
        for i in range(max_tool_turns)
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
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 30

    (tmp_path / "foo.py").write_text("# code")

    # Enough turns to exhaust all nudges: (max_plan_nudges + 1) * max_plan_investigation_turns
    n_turns = (cfg.max_plan_nudges + 1) * cfg.max_plan_investigation_turns + 1
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


# -- Phase tracking and budget tests (bench/yolo) ----------------------------

async def test_agent_phase_transitions(tmp_path):
    """Phase should transition: explore -> fix (on edit) -> verify (on run_command)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10

    (tmp_path / "foo.py").write_text("old line\n")

    fake = FakeClient(
        script=[
            # Turn 1: read (explore phase)
            CompletionResult(tool_calls=[_tc("read_file", {"path": "foo.py"})]),
            # Turn 2: edit (transitions to fix)
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "old line", "new_str": "new line"})
            ]),
            # Turn 3: run test (transitions to verify)
            CompletionResult(tool_calls=[
                _tc("run_command", {"command": "echo ok"}, call_id="c3")
            ]),
            # Turn 4: finish
            CompletionResult(text="Fixed.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert result.success
    assert result.final_text == "Fixed."


async def test_agent_force_explore_to_fix(tmp_path):
    """After max_explore_turns, a nudge should be injected in bench mode."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 20
    cfg.max_explore_turns = 3

    # Create files so reads succeed
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text(f"content {i}")

    script = [
        CompletionResult(tool_calls=[_tc("read_file", {"path": f"file{i}.py"}, call_id=f"c{i}")])
        for i in range(5)
    ] + [CompletionResult(text="giving up.", tool_calls=[])]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("explore a lot")

    assert result.success
    # A nudge about exploring too long should have been injected
    nudge_msgs = [
        m for m in result.messages
        if m.get("role") == "user" and "MUST call `edit_file`" in (m.get("content") or "")
    ]
    assert nudge_msgs, "expected an explore-to-fix nudge"


async def test_agent_caps_fix_verify_cycles(tmp_path):
    """Agent should force-finish after max_fix_verify_cycles edit->run cycles."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 50
    cfg.max_fix_verify_cycles = 3
    cfg.max_quality_retries = 100  # disable quality gate force-finish

    (tmp_path / "foo.py").write_text("line\n")

    # Build alternating edit + run_command pairs that always fail the test.
    script = []
    for i in range(10):
        # Reset the file so edits can succeed each time
        (tmp_path / "foo.py").write_text(f"line{i}\n")
        script.append(CompletionResult(tool_calls=[
            _tc("edit_file", {"path": "foo.py", "old_str": f"line{i}", "new_str": f"line{i+1}"},
                 call_id=f"e{i}")
        ]))
        script.append(CompletionResult(tool_calls=[
            _tc("run_command", {"command": f"python -m pytest test_{i}.py"}, call_id=f"r{i}")
        ]))
    script.append(CompletionResult(text="should not reach", tool_calls=[]))

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert result.success
    assert "cycle budget" in result.final_text


async def test_agent_force_finishes_after_test_pass(tmp_path):
    """After test passes post-edit, agent should force-finish within 2 turns."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 20

    (tmp_path / "foo.py").write_text("old\n")
    # Create a trivial test that always passes.
    (tmp_path / "test_foo.py").write_text("def test_ok(): pass\n")

    fake = FakeClient(
        script=[
            # Turn 1: edit
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "old", "new_str": "new"})
            ]),
            # Turn 2: run test (passes -> test_passed_after_edit, finish_countdown=2)
            CompletionResult(tool_calls=[
                _tc("run_command", {"command": "python -m pytest test_foo.py"}, call_id="c2")
            ]),
            # Turn 3: agent ignores nudge and reads (countdown 2->1)
            CompletionResult(tool_calls=[
                _tc("read_file", {"path": "foo.py"}, call_id="c3")
            ]),
            # Turn 4: agent ignores nudge again (countdown 1->0)
            CompletionResult(tool_calls=[
                _tc("read_file", {"path": "foo.py"}, call_id="c4")
            ]),
            # Turn 5: countdown hits 0 -> force finish before this turn runs
            CompletionResult(text="should not reach", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert result.success
    assert "did not stop after test passed" in result.final_text


async def test_post_edit_read_blocking(tmp_path):
    """After edits, consecutive read-only turns should trigger warning then blocking."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 30
    cfg.max_post_edit_read_turns = 2  # warn at 2, block at 4
    cfg.max_stuck_turns = 20  # disable stuck detection for this test
    cfg.max_history_messages = 50  # preserve all messages for assertion

    (tmp_path / "foo.py").write_text("old\n")
    for i in range(10):
        (tmp_path / f"r{i}.py").write_text(f"content {i}")

    script = [
        # Turn 1: edit (enters fix phase)
        CompletionResult(tool_calls=[
            _tc("edit_file", {"path": "foo.py", "old_str": "old", "new_str": "new"})
        ]),
        # Turns 2-7: read-only (post_edit_read_turns goes 1,2,3,4,5,6)
    ] + [
        CompletionResult(tool_calls=[
            _tc("read_file", {"path": f"r{i}.py"}, call_id=f"r{i}")
        ])
        for i in range(8)
    ] + [CompletionResult(text="done.", tool_calls=[])]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert result.success
    # Warning should appear at post_edit_read_turns == max_post_edit_read_turns
    warning_msgs = [
        m for m in result.messages
        if m.get("role") == "user" and "only reading files after making edits" in (m.get("content") or "")
    ]
    assert warning_msgs, "expected a post-edit read warning"
    # Blocking should appear at post_edit_read_turns >= max_post_edit_read_turns + 2
    blocked_msgs = [
        m for m in result.messages
        if m.get("role") == "tool" and "Exploration blocked" in (m.get("content") or "")
    ]
    assert blocked_msgs, "expected exploration to be blocked after too many read-only turns"


# -- Goal drift and edit failure tracking tests ---------------------------

async def test_goal_drift_detection(tmp_path):
    """Agent should get a GOAL DRIFT WARNING when editing unrelated files
    and encountering environmental errors."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 15
    cfg.max_stuck_turns = 20  # disable stuck nudges
    cfg.max_history_messages = 50

    # Create files that the agent will "fix" (unrelated to the problem).
    (tmp_path / "basic.py").write_text("from collections import Mapping\n")
    (tmp_path / "containers.py").write_text("from collections import MutableSet\n")

    # The user message mentions "separable.py" so basic.py/containers.py are "unrelated".
    problem_prompt = (
        "## Problem\n"
        "Bug in astropy/modeling/separable.py: nested CompoundModels wrong.\n"
    )

    fake = FakeClient(
        script=[
            # Turn 1: run test -> ImportError (env error #1)
            CompletionResult(tool_calls=[
                _tc("run_command", {"command": "python -m pytest"}, call_id="r1")
            ]),
            # Turn 2: edit basic.py (unrelated file #1)
            CompletionResult(tool_calls=[
                _tc("edit_file", {
                    "path": "basic.py",
                    "old_str": "from collections import Mapping",
                    "new_str": "from collections.abc import Mapping",
                })
            ]),
            # Turn 3: run test -> ImportError again (env error #2)
            CompletionResult(tool_calls=[
                _tc("run_command", {"command": "python -m pytest test_sep.py"}, call_id="r3")
            ]),
            # Turn 4: edit containers.py (unrelated file #2) -> should trigger drift
            CompletionResult(tool_calls=[
                _tc("edit_file", {
                    "path": "containers.py",
                    "old_str": "from collections import MutableSet",
                    "new_str": "from collections.abc import MutableSet",
                })
            ]),
            # Turn 5: finish
            CompletionResult(text="Fixed imports.", tool_calls=[]),
        ]
    )

    # Patch run_command to return ImportError in stderr
    original_run_tool = Agent._run_tool

    async def _mock_run_tool(self, turn, tc):
        if tc.name == "run_command":
            self._append_tool_result(
                tc, message='{"success": false, "error": "ImportError: cannot import name Mapping"}'
            )
            return {
                "success": False,
                "plan_approved": False,
                "data": {
                    "exit_code": 1,
                    "stderr": "ImportError: cannot import name 'Mapping' from 'collections'",
                    "stdout": "",
                },
            }
        return await original_run_tool(self, turn, tc)

    Agent._run_tool = _mock_run_tool  # type: ignore[assignment]
    try:
        agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
        result = await agent.run(problem_prompt)
    finally:
        Agent._run_tool = original_run_tool  # type: ignore[assignment]

    drift_msgs = [
        m for m in result.messages
        if m.get("role") == "user" and "GOAL DRIFT WARNING" in (m.get("content") or "")
    ]
    assert drift_msgs, "expected a goal drift warning when editing unrelated files"


async def test_edit_failure_nudge_at_3(tmp_path):
    """After 3 failed edits to the same file, a nudge should appear."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10
    cfg.max_stuck_turns = 20
    cfg.max_history_messages = 50

    (tmp_path / "foo.py").write_text("actual content here\n")

    fake = FakeClient(
        script=[
            # 3 failed edits (wrong old_str) then give up
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "wrong1", "new_str": "fix1"},
                     call_id="e1")
            ]),
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "wrong2", "new_str": "fix2"},
                     call_id="e2")
            ]),
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "wrong3", "new_str": "fix3"},
                     call_id="e3")
            ]),
            CompletionResult(text="giving up.", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix foo.py")

    nudge_msgs = [
        m for m in result.messages
        if m.get("role") == "user"
        and "failed to edit" in (m.get("content") or "")
        and "STOP guessing" in (m.get("content") or "")
    ]
    assert nudge_msgs, "expected a nudge after 3 failed edits to the same file"


async def test_problem_reanchor_at_turn_15(tmp_path):
    """After 15 turns, the problem statement should be re-injected."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 25
    cfg.max_stuck_turns = 20  # disable stuck nudges
    cfg.max_history_messages = 50

    for i in range(20):
        (tmp_path / f"f{i}.py").write_text(f"content {i}")

    # 16 turns of reading different files, then finish
    script = [
        CompletionResult(tool_calls=[
            _tc("read_file", {"path": f"f{i}.py"}, call_id=f"c{i}")
        ])
        for i in range(16)
    ] + [CompletionResult(text="done.", tool_calls=[])]

    problem_prompt = (
        "## Workflow\n1. Fix the bug.\n\n"
        "## Problem\nBug in foo/bar.py: the frobnicate function returns None.\n"
        "## Hints\nCheck the return statement.\n"
    )

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run(problem_prompt)

    reanchor_msgs = [
        m for m in result.messages
        if m.get("role") == "user"
        and "REMINDER" in (m.get("content") or "")
        and "frobnicate" in (m.get("content") or "")
    ]
    assert reanchor_msgs, "expected a problem re-anchoring message after turn 15"


async def test_consecutive_identical_loop_detection(tmp_path):
    """Agent force-finishes after 8 consecutive identical tool calls (bench)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 20

    (tmp_path / "foo.py").write_text("x = 1\n")

    # 15 identical read_file calls — should trigger loop detection at call 8.
    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id=f"c{i}")]
        )
        for i in range(15)
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert not result.success
    assert "loop detected" in result.error
    assert result.turns_used <= 9  # should stop well before 20


async def test_no_edit_force_finish_at_50_turns(tmp_path):
    """Agent force-finishes after 50 turns with no edits in bench mode."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 60

    (tmp_path / "foo.py").write_text("x = 1\n")

    # 55 turns alternating between read_file and search_files (not identical).
    script = [
        CompletionResult(
            tool_calls=[_tc(
                "read_file" if i % 2 == 0 else "search_files",
                {"path": "foo.py"} if i % 2 == 0 else {"pattern": f"pat{i}", "path": "."},
                call_id=f"c{i}",
            )]
        )
        for i in range(55)
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert not result.success
    assert "no edits after" in result.error
    assert result.turns_used == 50


async def test_turn_log_populated_in_bench(tmp_path):
    """TaskResult.turn_log is populated with per-turn events in bench mode."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 5

    (tmp_path / "foo.py").write_text("x = 1\n")

    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"})]
        ),
        CompletionResult(text="done."),
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert result.turn_log
    entry = result.turn_log[0]
    assert entry["turn"] == 1
    assert entry["phase"] == "explore"
    assert "tools" in entry
    assert entry["tools"][0]["name"] == "read_file"


async def test_task_result_has_phase_diagnostics(tmp_path):
    """TaskResult includes final_phase, explore_turns, fix_verify_cycles."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10

    (tmp_path / "foo.py").write_text("x = 1\n")
    (tmp_path / "test_foo.py").write_text("pass\n")

    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id="c1")]
        ),
        CompletionResult(
            tool_calls=[_tc("edit_file", {"path": "foo.py", "old_str": "x = 1", "new_str": "x = 2"}, call_id="c2")]
        ),
        CompletionResult(
            tool_calls=[_tc("run_command", {"command": "python -m pytest test_foo.py"}, call_id="c3")]
        ),
        CompletionResult(text="Fixed."),
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert result.success
    assert result.explore_turns >= 1
    assert result.final_phase in ("fix", "verify")
