"""End-to-end tests for plan mode workflow and plan state tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display
from squishy.plan_state import load_plan, plan_path

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


async def test_plan_mode_step_progress_tracking(tmp_path):
    """Verify step status transitions are tracked in plan state."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 6

    # Create test file
    (tmp_path / "app.py").write_text("# app\n")

    script = [
        CompletionResult(
            tool_calls=[
                _tc(
                    "plan_task",
                    {
                        "problem": "Multi-step task",
                        "solution": "Do several things",
                        "steps": ["Step 1", "Step 2"],
                    },
                )
            ]
        ),
        # After plan_task, agent gets system prompt about unresolved steps
        # Provide update_plan calls to complete the steps
        CompletionResult(
            tool_calls=[_tc("update_plan", {"step_index": 1, "status": "done"})]
        ),
        CompletionResult(
            tool_calls=[_tc("update_plan", {"step_index": 2, "status": "done"})]
        ),
        CompletionResult(text="Finished", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Do multi-step task")

    assert result.success

    # Verify plan state
    plan = load_plan(tmp_path)
    assert plan is not None
    assert len(plan.steps) == 2

    # Check step statuses
    assert plan.steps[0].status == "done"
    assert plan.steps[1].status == "done"

    # Check progress tracking
    assert result.plan_state is not None
    assert result.plan_state["progress"]["done"] == 2


async def test_plan_mode_files_created_tracking(tmp_path):
    """Verify files_created list is populated."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 5

    script = [
        CompletionResult(
            tool_calls=[
                _tc("write_file", {"path": "new.py", "content": "# new\n"}, call_id="c1")
            ]
        ),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Create new file")

    assert result.success

    # Verify files_created tracking
    utils_path = tmp_path / "new.py"
    assert utils_path.is_file()
    assert "new.py" in result.files_created


async def test_plan_mode_files_edited_tracking(tmp_path):
    """Verify files_edited list is populated for edits."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 6

    # Create initial file
    (tmp_path / "app.py").write_text("x = 1\n")

    script = [
        CompletionResult(tool_calls=[_tc("edit_file", {"path": "app.py", "old_str": "x = 1", "new_str": "x = 2"})]),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Edit app.py")

    assert result.success

    # Verify file was edited
    content = (tmp_path / "app.py").read_text()
    assert "x = 2" in content

    # Verify files_edited tracking
    assert result.files_edited == ["app.py"]


async def test_plan_mode_multiple_files_tracked(tmp_path):
    """Verify multiple file operations are tracked correctly."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 8

    script = [
        CompletionResult(
            tool_calls=[
                _tc("write_file", {"path": "new.py", "content": "# new\n"}, call_id="c1")
            ]
        ),
        CompletionResult(
            tool_calls=[
                _tc("write_file", {"path": "another.py", "content": "# another\n"}, call_id="c2")
            ]
        ),
        CompletionResult(
            tool_calls=[
                _tc("edit_file", {"path": "new.py", "old_str": "# new\n", "new_str": "# modified\n"}, call_id="c3")
            ]
        ),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Create and modify files")

    assert result.success
    assert "new.py" in result.files_created
    assert "another.py" in result.files_created
    # new.py appears in files_edited too (edited after creation)
    assert "new.py" in result.files_edited


async def test_plan_state_persistence_across_turns(tmp_path):
    """Plan state is persisted after plan_task call in edits mode."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 10

    script = [
        CompletionResult(
            tool_calls=[
                _tc(
                    "plan_task",
                    {
                        "problem": "Task 1",
                        "solution": "Solution 1",
                        "steps": ["Step A", "Step B"],
                    },
                )
            ]
        ),
        # After plan_task, agent gets system prompt about unresolved steps
        # Provide update_plan calls to complete the steps
        CompletionResult(
            tool_calls=[_tc("update_plan", {"step_index": 1, "status": "done"})]
        ),
        CompletionResult(
            tool_calls=[_tc("update_plan", {"step_index": 2, "status": "done"})]
        ),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Do task")

    assert result.success

    # Plan should be persisted to disk (approved by default in edits mode)
    plan = load_plan(tmp_path)
    assert plan is not None
    assert len(plan.steps) == 2
    assert plan.problem == "Task 1"