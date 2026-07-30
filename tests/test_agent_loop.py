"""Agent loop tests using a scripted fake Client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display
from squishy.plan_state import plan_path

from conftest import FakeClient
 
 
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
            on_retry: Any = None,
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
    assert any("call `plan_task` now" in (m.get("content") or "") for m in nudge_msgs)


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


async def test_json_plan_in_prose_triggers_pointed_nudge(tmp_path):
    """When the model writes the plan_task fields as JSON in prose, the
    agent should nudge with the 'use the tool, not prose' wording so the
    model self-corrects faster."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 4
    cfg.max_plan_nudges = 2

    json_plan = (
        'Here is my plan:\n'
        '{ "problem": "p", "solution": "s", "steps": ["a", "b"] }'
    )
    fake = FakeClient(
        script=[CompletionResult(text=json_plan, tool_calls=[]) for _ in range(5)]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    await agent.run("plan please")

    nudges = [
        m["content"] for m in agent.messages
        if m.get("role") == "user" and str(m.get("content", "")).startswith("[system]")
    ]
    assert any("JSON in prose is not a plan" in n for n in nudges)


async def test_ctrl_c_at_plan_approval_cancels_run(tmp_path):
    """Ctrl+C at the approval prompt must abort the entire turn rather
    than being silently downgraded to a 'decline'."""
    from squishy.errors import AgentCancelled
    from squishy.plan_state import plan_path

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
                        {"problem": "p", "solution": "s", "steps": ["a"]},
                    )
                ]
            )
        ]
    )

    async def hostile_prompt(_tool, _args):
        raise KeyboardInterrupt

    agent = Agent(cfg, fake, Display(), prompt_fn=hostile_prompt)  # type: ignore[arg-type]
    with pytest.raises(AgentCancelled):
        await agent.run("plan please")

    # The persisted plan must be cleared so the next run starts fresh.
    assert not plan_path(tmp_path).exists()


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

    # Create files so read_file succeeds (different files to avoid quality gate)
    for i in range(max_tool_turns):
        (tmp_path / f"file{i}.py").write_text(f"# code {i}")

    # max_plan_investigation_turns turns of read-only tool calls, then plan_task
    script = [
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": f"file{i}.py"}, call_id=f"c{i}")]
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
    # Either path should kill the run: the plan-task nudge budget OR the
    # cross-mode loop detector. Both indicate the model never produced a plan.
    assert "plan_task" in result.error or "loop detected" in result.error


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
    """After max_explore_turns, a phase transition should move to plan in bench mode."""
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
    # Phase transition notification should have been injected
    nudge_msgs = [
        m for m in result.messages
        if m.get("role") == "user" and "Phase:" in (m.get("content") or "")
    ]
    assert nudge_msgs, "expected a phase transition notification"


async def test_agent_force_finishes_after_test_pass(tmp_path):
    """After test passes post-edit, agent should force-finish via done phase."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 20
    cfg.max_explore_turns = 1  # fast explore->plan transition
    cfg.max_plan_turns = 1    # fast plan->execute transition

    (tmp_path / "foo.py").write_text("old\n")
    # Create a trivial test that always passes.
    (tmp_path / "test_foo.py").write_text("def test_ok(): pass\n")

    fake = FakeClient(
        script=[
            # Turn 1: read (explore phase, explore_turns=1 -> plan transition)
            CompletionResult(tool_calls=[
                _tc("read_file", {"path": "foo.py"})
            ]),
            # Turn 2: save_note (plan phase, plan_turns=1 -> execute transition)
            CompletionResult(tool_calls=[
                _tc("save_note", {"key": "fix", "content": "change old to new"}, call_id="c2")
            ]),
            # Turn 3: edit (execute phase, has_edit=True)
            CompletionResult(tool_calls=[
                _tc("edit_file", {"path": "foo.py", "old_str": "old", "new_str": "new"}, call_id="c3")
            ]),
            # Turn 4: run test (passes -> execute->verify, test_passed_after_edit -> done)
            CompletionResult(tool_calls=[
                _tc("run_command", {"command": "python -m pytest test_foo.py"}, call_id="c4")
            ]),
            # Should not reach turn 5 — done phase force-finishes at top of loop
            CompletionResult(text="should not reach", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix it")

    assert result.success
    assert "Fix applied" in result.final_text


# -- Goal drift and edit failure tracking tests ---------------------------

async def test_goal_drift_detection(tmp_path):
    """Agent should get a goal drift nudge when editing unrelated files
    and encountering environmental errors."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 15
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

    # Patch run_tool as seen by agent.py (imported by name into its namespace).
    from unittest.mock import patch
    from squishy.agent_dispatch import run_tool as _original_run_tool, append_tool_result

    async def _mock_run_tool(agent_obj, turn, tc):
        if tc.name == "run_command":
            append_tool_result(
                agent_obj, tc,
                message='{"success": false, "error": "ImportError: cannot import name Mapping"}',
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
        return await _original_run_tool(agent_obj, turn, tc)

    with patch("squishy.agent.run_tool", _mock_run_tool):
        agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
        result = await agent.run(problem_prompt)

    drift_msgs = [
        m for m in result.messages
        if m.get("role") == "user"
        and "not the reported bug" in (m.get("content") or "")
    ]
    assert drift_msgs, "expected a goal drift nudge when editing unrelated files"


async def test_edit_failure_nudge_at_3(tmp_path):
    """After 3 failed edits to the same file, a nudge should appear."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 10
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
        and "failed edits" in (m.get("content") or "")
    ]
    assert nudge_msgs, "expected a nudge after 3 failed edits to the same file"


async def test_problem_reanchor_at_turn_15(tmp_path):
    """After 15 turns, the problem statement should be re-injected."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 25
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
    """Agent force-finishes early when stuck without edits (quality gate or max turns)."""
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
    # Quality gate or max turns should stop the agent.
    assert result.error  # some error message should be set


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
    cfg.max_explore_turns = 1  # fast explore->plan transition
    cfg.max_plan_turns = 1    # fast plan->execute transition

    (tmp_path / "foo.py").write_text("x = 1\n")
    (tmp_path / "test_foo.py").write_text("def test_ok(): pass\n")

    script = [
        # Turn 1 (explore): read file → explore_turns=1 → plan
        CompletionResult(
            tool_calls=[_tc("read_file", {"path": "foo.py"}, call_id="c1")]
        ),
        # Turn 2 (plan): save_note → plan_turns=1 → execute
        CompletionResult(
            tool_calls=[_tc("save_note", {"key": "fix", "content": "change x"}, call_id="c2")]
        ),
        # Turn 3 (execute): edit file → has_edit=True
        CompletionResult(
            tool_calls=[_tc("edit_file", {"path": "foo.py", "old_str": "x = 1", "new_str": "x = 2"}, call_id="c3")]
        ),
        # Turn 4 (execute): run test → test passes → done
        CompletionResult(
            tool_calls=[_tc("run_command", {"command": "python -m pytest test_foo.py"}, call_id="c4")]
        ),
        CompletionResult(text="Fixed."),
    ]
    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix the bug")

    assert result.success
    assert result.explore_turns >= 1
    assert result.final_phase in ("execute", "verify", "done")


# ---------------------------------------------------------------------------
# Plan rejection with feedback
# ---------------------------------------------------------------------------


async def test_plan_rejection_without_feedback(tmp_path):
    """Rejecting a plan with no feedback gives a generic decline message."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5

    fake = FakeClient(
        script=[
            # Turn 1: propose a plan — will be rejected
            CompletionResult(tool_calls=[
                _tc("plan_task", {
                    "plan": "Bad plan",
                    "problem": "Bug in foo",
                    "solution": "Delete everything",
                    "steps": ["step 1"],
                })
            ]),
            # Turn 2: agent gives up after rejection
            CompletionResult(text="Understood, what would you like changed?"),
        ],
    )

    async def reject(_tool, _args):
        return False

    agent = Agent(cfg, fake, Display(), prompt_fn=reject)  # type: ignore[arg-type]
    result = await agent.run("plan something")

    # Plan was cleared
    assert agent.tool_ctx.plan is None
    # Agent saw the generic decline message and continued
    assert fake._i == 2
    # Check the tool result message contains the generic decline
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    decline_msg = tool_msgs[0]["content"]
    assert "declined" in decline_msg.lower()
    assert "feedback" not in decline_msg.lower()


async def test_plan_rejection_with_feedback(tmp_path):
    """Rejecting a plan with feedback passes the user's comments to the agent."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5

    feedback_text = "Use a safer approach, don't delete the database"

    fake = FakeClient(
        script=[
            # Turn 1: propose a plan — will be rejected with feedback
            CompletionResult(tool_calls=[
                _tc("plan_task", {
                    "plan": "Dangerous plan",
                    "problem": "Slow queries",
                    "solution": "Drop and recreate tables",
                    "steps": ["drop tables", "recreate"],
                })
            ]),
            # Turn 2: agent responds after seeing feedback
            CompletionResult(text="I'll revise the plan."),
        ],
    )

    async def reject_with_feedback(_tool, _args):
        return feedback_text

    agent = Agent(cfg, fake, Display(), prompt_fn=reject_with_feedback)  # type: ignore[arg-type]
    result = await agent.run("plan something")

    # Plan was cleared
    assert agent.tool_ctx.plan is None
    # Agent continued after rejection
    assert fake._i == 2
    # Check the tool result contains the user's feedback
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    decline_msg = tool_msgs[0]["content"]
    assert feedback_text in decline_msg
    assert "Revise the plan" in decline_msg


async def test_plan_rejection_then_revised_plan_approved(tmp_path):
    """Agent revises plan after rejection feedback, second plan is approved."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5

    call_count = 0

    async def reject_then_approve(_tool, _args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Add a backup step before modifying anything"
        return True

    fake = FakeClient(
        script=[
            # Turn 1: first plan — rejected with feedback
            CompletionResult(tool_calls=[
                _tc("plan_task", {
                    "plan": "Risky plan",
                    "problem": "Data migration",
                    "solution": "Migrate in-place",
                    "steps": ["modify schema"],
                })
            ]),
            # Turn 2: revised plan — approved
            CompletionResult(tool_calls=[
                _tc("plan_task", {
                    "plan": "Safe plan",
                    "problem": "Data migration",
                    "solution": "Backup then migrate",
                    "steps": ["backup database", "modify schema"],
                }, call_id="c2")
            ]),
        ],
    )

    agent = Agent(cfg, fake, Display(), prompt_fn=reject_then_approve)  # type: ignore[arg-type]
    result = await agent.run("plan a migration")

    assert result.success
    assert "Plan approved" in result.final_text
    assert agent.tool_ctx.plan is not None
    assert agent.tool_ctx.plan.approved is True
    # Both completions were consumed
    assert fake._i == 2
    assert call_count == 2


async def test_plan_approval_still_works(tmp_path):
    """Approval on first try still works with the updated prompt_fn signature."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 5

    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[
                _tc("plan_task", {
                    "plan": "Good plan",
                    "problem": "Add feature X",
                    "solution": "Implement it",
                    "steps": ["write code", "test"],
                })
            ]),
            CompletionResult(text="should not run"),
        ],
    )

    async def approve(_tool, _args):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=approve)  # type: ignore[arg-type]
    result = await agent.run("plan feature X")

    assert result.success
    assert "Plan approved" in result.final_text
    assert agent.tool_ctx.plan is not None
    assert agent.tool_ctx.plan.approved is True
    assert fake._i == 1


async def test_consecutive_identical_resets_after_edit(tmp_path):
    """A successful edit_file should reset the consecutive-identical counter so
    that subsequent identical calls start a fresh streak rather than continuing
    the old one.

    Scenario:
      - 4 identical read_file calls (streak = 3, below the 7-call hard limit but
        above the 2-call nudge threshold).
      - 1 successful edit_file call  → counter resets to 0.
      - 4 more identical read_file calls → streak reaches 3 again, not 7, so the
        agent does NOT force-finish with "loop detected".
    """
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 30

    (tmp_path / "foo.py").write_text("old line\n")
    # Provide multiple files so the quality gate (excessive_reread) doesn't fire.
    for i in range(10):
        (tmp_path / f"r{i}.py").write_text(f"content {i}")

    script = [
        # 4 identical read_file calls — streak hits 3 (nudge fires) but NOT >=7
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r0.py"}, call_id="a0")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r0.py"}, call_id="a1")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r0.py"}, call_id="a2")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r0.py"}, call_id="a3")]),
        # Successful edit — resets consecutive_identical to 0
        CompletionResult(tool_calls=[
            _tc("edit_file", {"path": "foo.py", "old_str": "old line", "new_str": "new line"})
        ]),
        # 4 more identical read_file calls — fresh streak from 0, should NOT trigger loop
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r1.py"}, call_id="b0")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r1.py"}, call_id="b1")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r1.py"}, call_id="b2")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "r1.py"}, call_id="b3")]),
        CompletionResult(text="done.", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("fix foo.py")

    # The agent should finish successfully (edit was made) — "loop detected" should NOT appear.
    assert "loop detected" not in (result.error or ""), (
        f"consecutive_identical counter was not reset after edit: {result.error}"
    )
    # The edit should have been applied.
    assert (tmp_path / "foo.py").read_text() == "new line\n"


async def test_recall_miss_relaxes_recall_enforcement(tmp_path):
    """When recall returns zero matches, plan-mode enforcement must relax so
    the model can fall back to reads without being nagged to 'use recall'."""
    from squishy.index import build_index, save_index

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 6
    cfg.max_recall_skip_turns = 2
    (tmp_path / "foo.py").write_text('"""Foo module."""\ndef widget(): return 1\n')
    save_index(str(tmp_path), build_index(str(tmp_path)))

    fake = FakeClient(
        script=[
            # recall for something not in the index -> total_matched == 0
            CompletionResult(tool_calls=[_tc("recall", {"query": "zzz_no_such_symbol"})]),
            CompletionResult(tool_calls=[_tc("read_file", {"path": "foo.py"})]),
            CompletionResult(tool_calls=[_tc("read_file", {"path": "foo.py", "offset": 1}, "c2")]),
            CompletionResult(tool_calls=[_tc("read_file", {"path": "foo.py", "offset": 2}, "c3")]),
            CompletionResult(tool_calls=[_tc("plan_task", {"problem": "p", "solution": "s", "steps": ["a"]})]),
        ]
    )

    async def auto_approve(_tool, _args):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    assert agent.has_index  # index was built, so enforcement is otherwise active
    result = await agent.run("find the widget")

    assert result.success
    assert agent.recall_missed
    assert not any(
        "Too many read calls without `recall`" in (m.get("content") or "")
        for m in result.messages
    ), "recall-miss should have relaxed the recall-first enforcement"


async def test_alias_tool_call_dispatches_canonically(tmp_path):
    """A model that emits `create`/`file_path` (another harness's vocabulary)
    should have it normalized and dispatched as write_file."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 4
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("create", {"file_path": "hi.py", "content": "print(1)\n"})]),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    result = await agent.run("make hi.py")
    assert result.success
    assert (tmp_path / "hi.py").read_text() == "print(1)\n"
    assert "hi.py" in result.files_created
    # The recorded assistant call was rewritten to the canonical tool name.
    tool_names = [
        tc["function"]["name"]
        for m in result.messages if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    ]
    assert "write_file" in tool_names
    assert "create" not in tool_names


async def test_user_message_is_persisted_to_session(tmp_path):
    """#1: the user turn must reach the session log even when the run ends via
    a nudge/continue path (previously silently dropped)."""
    from squishy.session import create_session, load_messages

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.session_dir = str(tmp_path / "sessions")
    cfg.max_turns = 4
    sess = create_session(model="fake", working_dir=str(tmp_path), mode="yolo",
                          tools=[], root=cfg.session_dir)
    fake = FakeClient(script=[CompletionResult(text="done", tool_calls=[])])
    agent = Agent(cfg, fake, display=None, session_id=sess.id)  # type: ignore[arg-type]
    await agent.run("REMEMBER-THIS-PROMPT")

    persisted = load_messages(sess.id, root=cfg.session_dir)
    user_msgs = [m for m in persisted if m.get("role") == "user"]
    assert any("REMEMBER-THIS-PROMPT" in (m.get("content") or "") for m in user_msgs)


async def test_result_messages_exclude_live_context_pair(tmp_path):
    """#11: the synthetic live-context pair must not leak into TaskResult."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 4
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[_tc("save_note", {"key": "k", "content": "v"})]),
            CompletionResult(text="done", tool_calls=[]),
        ]
    )
    agent = Agent(cfg, fake, display=None)  # type: ignore[arg-type]
    result = await agent.run("note something")
    assert not any(m.get("_squishy_live_ctx") for m in result.messages)
    assert not any(m.get("name") == "_squishy_context" for m in result.messages)
    assert not any(m.get("name") == "_squishy_context" for m in result.full_log)


async def test_recall_then_reads_does_not_trip_enforcement(tmp_path):
    """#10: the recommended recall→read→read pattern must not trigger the
    'too many reads without recall' nudge."""
    from squishy.index import build_index, save_index

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 6
    cfg.max_recall_skip_turns = 2
    (tmp_path / "foo.py").write_text('"""Foo."""\ndef widget(): return 1\n')
    save_index(str(tmp_path), build_index(str(tmp_path)))
    fake = FakeClient(
        script=[
            CompletionResult(tool_calls=[
                _tc("recall", {"query": "widget"}, "c1"),
                _tc("read_file", {"path": "foo.py"}, "c2"),
                _tc("read_file", {"path": "foo.py", "offset": 1}, "c3"),
            ]),
            CompletionResult(tool_calls=[_tc("plan_task", {"problem": "p", "solution": "s", "steps": ["a"]})]),
        ]
    )

    async def auto_approve(_t, _a):
        return True

    agent = Agent(cfg, fake, Display(), prompt_fn=auto_approve)  # type: ignore[arg-type]
    result = await agent.run("find widget")
    assert not any(
        "Too many read calls without `recall`" in (m.get("content") or "")
        for m in result.messages
    )


async def test_must_edit_gate_removes_run_command(tmp_path):
    """After max_turns_without_edit turns with no edit, run_command is pulled
    from the schema so the model can only read/edit (100% patch-rate goal)."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 8
    cfg.max_turns_without_edit = 3
    (tmp_path / "a.py").write_text("x = 1\n")

    seen_schemas: list[list[str]] = []

    class _Recorder(FakeClient):
        async def complete(self, messages, tools, **kw):
            seen_schemas.append([t["function"]["name"] for t in tools])
            return await super().complete(messages, tools, **kw)

    script = [
        CompletionResult(tool_calls=[_tc("run_command", {"command": "ls"}, f"c{i}")])
        for i in range(6)
    ] + [CompletionResult(text="done", tool_calls=[])]
    agent = Agent(cfg, _Recorder(script=script), display=None)  # type: ignore[arg-type]
    await agent.run("fix it")

    assert any("run_command" in s for s in seen_schemas), "should start available"
    assert not seen_schemas[-1].count("run_command"), "should be withdrawn after the budget"
    assert any("edit_file" in s for s in seen_schemas[-1:]), "editing stays available"


async def test_must_edit_gate_not_applied_after_an_edit(tmp_path):
    """A successful edit keeps run_command available for verification."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "yolo"
    cfg.max_turns = 8
    cfg.max_turns_without_edit = 2
    (tmp_path / "a.py").write_text("x = 1\n")

    seen: list[list[str]] = []

    class _Recorder(FakeClient):
        async def complete(self, messages, tools, **kw):
            seen.append([t["function"]["name"] for t in tools])
            return await super().complete(messages, tools, **kw)

    script = [
        CompletionResult(tool_calls=[_tc("edit_file", {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})]),
        CompletionResult(tool_calls=[_tc("run_command", {"command": "ls"}, "c2")]),
        CompletionResult(tool_calls=[_tc("run_command", {"command": "pwd"}, "c3")]),
        CompletionResult(tool_calls=[_tc("run_command", {"command": "echo hi"}, "c4")]),
        CompletionResult(text="done", tool_calls=[]),
    ]
    agent = Agent(cfg, _Recorder(script=script), display=None)  # type: ignore[arg-type]
    await agent.run("fix it")
    assert "run_command" in seen[-1], "run_command must remain after an edit landed"


async def test_must_edit_gate_forces_execute_phase_in_bench(tmp_path):
    """In bench mode the phase machine gates tools and explore/verify don't
    expose edit_file, so the gate must also move the phase to execute —
    otherwise withdrawing run_command leaves nothing actionable."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 8
    cfg.max_turns_without_edit = 2
    (tmp_path / "a.py").write_text("x = 1\n")

    seen: list[list[str]] = []

    class _Recorder(FakeClient):
        async def complete(self, messages, tools, **kw):
            seen.append([t["function"]["name"] for t in tools])
            return await super().complete(messages, tools, **kw)

    script = [
        CompletionResult(tool_calls=[_tc("read_file", {"path": "a.py"}, f"c{i}")])
        for i in range(6)
    ] + [CompletionResult(text="done", tool_calls=[])]
    agent = Agent(cfg, _Recorder(script=script), display=None)  # type: ignore[arg-type]
    await agent.run("fix it")

    late = seen[-1]
    assert "edit_file" in late, f"edit_file must be reachable once forced: {late}"
    assert "run_command" not in late, f"run_command should be withdrawn: {late}"
