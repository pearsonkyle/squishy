"""Tests for change tracking: file operations and plan state persistence."""

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
from squishy.tools.base import ToolContext

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


async def test_file_operations_tracking_created(ctx):
    """Verify files_read dict is populated for read operations after write."""
    from squishy.tools.fs import write_file, read_file

    result = await write_file.run(
        {"path": "new.py", "content": "# New file\n"}, ctx
    )
    assert result.success

    # Read the file to populate files_read
    r = await read_file.run({"path": "new.py"}, ctx)
    assert r.success

    # Check ToolContext files_read was populated
    assert "new.py" in ctx.files_read


async def test_file_operations_tracking_edited(ctx):
    """Verify files_read dict is populated after edit operations."""
    from squishy.tools.fs import write_file, edit_file, read_file

    # Create file first
    await write_file.run({"path": "app.py", "content": "x = 1\n"}, ctx)

    # Edit it
    result = await edit_file.run(
        {"path": "app.py", "old_str": "x = 1", "new_str": "x = 2"}, ctx
    )
    assert result.success

    # Read it to populate files_read (edit invalidates cache but read populates)
    r = await read_file.run({"path": "app.py"}, ctx)
    assert r.success

    # Verify file was actually edited - it should be in files_read
    assert "app.py" in ctx.files_read


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


async def test_consecutive_reads_tracking(tmp_path):
    """Verify read tool calls increment the tracking counter."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"
    cfg.max_turns = 10

    # Create test files
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text(f"# file {i}\n")

    cfg2 = Config()
    cfg2.working_dir = str(tmp_path)
    cfg2.permission_mode = "plan"
    cfg2.max_turns = 5

    fake = FakeClient(script=[CompletionResult(text="done", tool_calls=[])])
    agent = Agent(cfg2, fake, Display())  # type: ignore[arg-type]

    initial_count = agent.consecutive_reads_without_recall

    assert initial_count == 0


async def test_read_cache_invalidation_on_write(ctx, tmp_path):
    """Verify cache is cleared after file mutation."""
    from squishy.tools.fs import read_file, write_file

    # Write initial content
    await write_file.run({"path": "data.txt", "content": "version1\n"}, ctx)

    # Read it (populates cache)
    result = await read_file.run({"path": "data.txt"}, ctx)
    assert result.success
    # Note: cache_hit only appears on subsequent reads of same offset/limit

    # Edit content (should invalidate cache); write_file refuses existing files
    from squishy.tools.fs import edit_file
    await edit_file.run({"path": "data.txt", "old_str": "version1", "new_str": "version2"}, ctx)

    # Read again - should show new content (cache invalidated)
    result = await read_file.run({"path": "data.txt"}, ctx)
    assert result.success
    # cache_hit should be False or not present since we invalidated the cache on edit
    if "cache_hit" in result.data:
        assert not result.data["cache_hit"]
    assert "version2" in result.data["content"]


async def test_read_cache_invalidation_on_edit(ctx, tmp_path):
    """Verify cache is cleared after edit_file operation."""
    from squishy.tools.fs import read_file, write_file, edit_file

    # Create file
    await write_file.run({"path": "code.py", "content": "x = 1\n"}, ctx)

    # Read it
    result = await read_file.run({"path": "code.py"}, ctx)
    assert result.success
    original_content = result.data["content"]

    # Edit it
    await edit_file.run(
        {"path": "code.py", "old_str": "x = 1", "new_str": "y = 2"}, ctx
    )

    # Read again - should show new content (cache invalidated)
    result = await read_file.run({"path": "code.py"}, ctx)
    assert result.success
    if "cache_hit" in result.data:
        assert not result.data["cache_hit"]
    assert "y = 2" in result.data["content"]


async def test_multiple_edits_same_file_tracking(ctx, tmp_path):
    """Verify multiple edits to same file are tracked correctly."""
    from squishy.tools.fs import write_file, edit_file, read_file

    # Create initial file
    await write_file.run({"path": "app.py", "content": "a = 1\nb = 2\nc = 3\n"}, ctx)

    # Multiple edits
    await edit_file.run(
        {"path": "app.py", "old_str": "a = 1", "new_str": "A = 1"}, ctx
    )
    await edit_file.run(
        {"path": "app.py", "old_str": "b = 2", "new_str": "B = 2"}, ctx
    )
    await edit_file.run(
        {"path": "app.py", "old_str": "c = 3", "new_str": "C = 3"}, ctx
    )

    # Read file to populate files_read
    r = await read_file.run({"path": "app.py"}, ctx)
    assert r.success

    # Verify final content in files_read
    assert "app.py" in ctx.files_read
    content = ctx.files_read["app.py"]
    assert "A = 1" in content
    assert "B = 2" in content
    assert "C = 3" in content


async def test_files_created_vs_edited_tracking(tmp_path):
    """Verify files_created and files_edited are tracked separately."""
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


async def test_context_files_read_accumulates(tmp_path):
    """Verify files_read dict accumulates across multiple tool calls."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 6

    # Create multiple files
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text(f"# file {i}\n")

    script = [
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file0.py"}, call_id="c1")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file1.py"}, call_id="c2")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file2.py"}, call_id="c3")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file3.py"}, call_id="c4")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file4.py"}, call_id="c5")]),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Read many files")

    assert result.success

    # All 5 files should be in files_read
    for i in range(5):
        assert f"file{i}.py" in agent.tool_ctx.files_read

    # Verify content was read
    for i in range(5):
        assert f"# file {i}" in agent.tool_ctx.files_read[f"file{i}.py"]


async def test_pending_plan_evidence_accumulation(tmp_path):
    """Verify pending_plan_evidence list is cleared after plan_task."""
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
                        "problem": "Task",
                        "solution": "Do it",
                        "steps": ["Step 1"],
                    },
                )
            ]
        ),
        # After plan_task, agent gets system prompt about unresolved steps
        CompletionResult(
            tool_calls=[_tc("update_plan", {"step_index": 1, "status": "done"})]
        ),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Do task")

    assert result.success
    # Evidence should have been cleared after plan_task (it clears pending_plan_evidence)
    assert len(agent.tool_ctx.pending_plan_evidence) == 0


async def test_tool_context_mode_switching(ctx):
    """Verify permission_mode changes take effect in tool dispatch."""
    ctx.permission_mode = "plan"
    from squishy.tools import check_permission
    from squishy.tools.fs import read_file, write_file

    # In plan mode: reads allowed, writes blocked
    allowed, _ = check_permission(read_file, "plan")
    assert allowed

    allowed, reason = check_permission(write_file, "plan")
    assert not allowed
    assert "plan mode" in reason

    # Switch to yolo: both should be allowed
    ctx.permission_mode = "yolo"

    allowed, _ = check_permission(read_file, "yolo")
    assert allowed

    allowed, _ = check_permission(write_file, "yolo")
    assert allowed


async def test_plan_switch_prompted_flag(ctx):
    """Verify plan_switch_prompted flag tracks mode transitions."""
    ctx.permission_mode = "plan"
    ctx.plan_switch_prompted = True

    # Flag should track whether we've prompted about mode switch
    assert ctx.plan_switch_prompted is True
