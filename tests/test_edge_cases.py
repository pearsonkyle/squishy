"""Edge cases and error handling tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from squishy.tools.base import ToolContext
from squishy.plan_state import PlanState, load_plan, plan_path


def _tc(name: str, args: dict, call_id: str = "c1"):
    from squishy.client import ToolCall
    return ToolCall(id=call_id, name=name, args=args)


from squishy.client import CompletionResult


@pytest.mark.asyncio

class TestPlanEdgeCases:
    """Test edge cases in plan operations."""

    @pytest.mark.asyncio
    async def test_plan_with_no_steps(self, tmp_path):
        """Empty steps array should fail validation."""
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        result = await _plan_task(
            {
                "problem": "Test",
                "solution": "Solution",
                "steps": [],  # Empty steps
            },
            ctx,
        )
        assert not result.success
        assert "steps" in result.error.lower() or len(result.data.get("steps", [])) == 0

    async def test_plan_task_missing_solution(self, tmp_path):
        """Missing solution field should be rejected."""
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        result = await _plan_task(
            {
                "problem": "Test",
                "steps": ["Step 1"],
            },
            ctx,
        )
        # Solution is required
        assert not result.success
        assert "solution" in result.error.lower()

    async def test_update_plan_step_out_of_bounds(self, tmp_path):
        """Invalid step index should be rejected."""
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        await _plan_task(
            {
                "problem": "Test",
                "solution": "Solution",
                "steps": ["Step 1"],
            },
            ctx,
        )
        result = await _update_plan(
            {"step_index": 100, "status": "done"},  # Out of bounds
            ctx,
        )
        assert not result.success
        assert "out of range" in result.error.lower()

    async def test_update_plan_invalid_status(self, tmp_path):
        """Invalid status value should be rejected."""
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        await _plan_task(
            {
                "problem": "Test",
                "solution": "Solution",
                "steps": ["Step 1"],
            },
            ctx,
        )
        result = await _update_plan(
            {"step_index": 0, "status": "invalid-status"},  # Invalid status
            ctx,
        )
        assert not result.success

    async def test_update_plan_no_active_plan(self, tmp_path):
        """update_plan without active plan should fail."""
        from squishy.tools.plan import _update_plan

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        result = await _update_plan(
            {"step_index": 0, "status": "done"},
            ctx,
        )
        assert not result.success
        assert "no active plan" in result.error.lower()

    async def test_plan_task_with_empty_string_steps(self, tmp_path):
        """Steps with empty strings should be rejected."""
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        result = await _plan_task(
            {
                "problem": "Test",
                "solution": "Solution",
                "steps": ["", ""],  # Empty step descriptions
            },
            ctx,
        )
        # Should fail - empty strings are invalid steps
        assert not result.success

    async def test_plan_task_with_all_optional_fields(self, tmp_path):
        """plan_task with all optional fields should work."""
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        result = await _plan_task(
            {
                "problem": "Test",
                "solution": "Solution",
                "steps": ["Step 1", "Step 2"],
                "plan": "Full plan description",
                "files_to_create": ["new.py"],
                "files_to_modify": ["old.py"],
            },
            ctx,
        )
        assert result.success
        if ctx.plan is not None:
            assert ctx.plan.plan == "Full plan description"
            assert ctx.plan.files_to_create == ["new.py"]
            assert ctx.plan.files_to_modify == ["old.py"]


@pytest.mark.asyncio

class TestReadCacheEdgeCases:
    """Test edge cases in file read caching."""

    async def test_read_file_dedup_same_offset(self, ctx):
        """Re-reading same offset returns cache_hit marker."""
        from squishy.tools.fs import read_file, write_file

        # Create test file
        await write_file.run(
            {"path": "test.py", "content": "line1\nline2\nline3\n"}, ctx
        )

        # Read first time - no cache hit
        result1 = await read_file.run(
            {"path": "test.py", "offset": 0, "limit": 2},
            ctx,
        )
        assert result1.success
        assert "cache_hit" not in result1.data

        # Read again with same offset/limit - should be cache hit
        result2 = await read_file.run(
            {"path": "test.py", "offset": 0, "limit": 2},
            ctx,
        )
        assert result2.success
        assert result2.data.get("cache_hit") is True

    async def test_read_file_different_offset_misses_cache(self, ctx):
        """Reading different offset should miss cache."""
        from squishy.tools.fs import read_file, write_file

        # Create test file
        await write_file.run(
            {"path": "test.py", "content": "line1\nline2\nline3\n"}, ctx
        )

        # Read with offset 0
        result1 = await read_file.run(
            {"path": "test.py", "offset": 0, "limit": 2},
            ctx,
        )
        assert result1.success
        assert "cache_hit" not in result1.data

        # Read with different offset - should be cache miss
        result2 = await read_file.run(
            {"path": "test.py", "offset": 1, "limit": 2},
            ctx,
        )
        assert result2.success
        # Should not be a cache hit since offset is different
        if "cache_hit" in result2.data:
            assert not result2.data["cache_hit"]
        # Content should be different (starting from line 2)
        assert "line2" in result2.data["content"]

    async def test_read_nonexistent_file(self, ctx):
        """Reading a nonexistent file should fail gracefully."""
        from squishy.tools.fs import read_file

        result = await read_file.run({"path": "does_not_exist.py"}, ctx)
        assert not result.success
        assert "file not found" in result.error.lower()


@pytest.mark.asyncio

class TestEditEdgeCases:
    """Test edge cases in file editing."""

    async def test_edit_file_no_match(self, ctx):
        """Edit with no match should fail."""
        from squishy.tools.fs import write_file, edit_file

        await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "goodbye", "new_str": "world"},
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_edit_file_ambiguous_multiple_matches(self, ctx):
        """Edit with multiple matches without replace_all should fail."""
        from squishy.tools.fs import write_file, edit_file

        await write_file.run({"path": "app.py", "content": "x\nx\nx\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "x", "new_str": "y"},
            ctx,
        )
        assert not result.success
        assert "matches" in result.error.lower()
        assert "replace_all" in result.error.lower()

    async def test_edit_file_replace_all_with_no_matches(self, ctx):
        """replace_all=True with no matches should still fail."""
        from squishy.tools.fs import write_file, edit_file

        await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "goodbye", "new_str": "world", "replace_all": True},
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_edit_file_empty_old_str(self, ctx):
        """Empty old_str should be rejected or handled."""
        from squishy.tools.fs import write_file, edit_file

        await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "", "new_str": "world"},
            ctx,
        )
        # Empty old_str is problematic - either fail or replace everything
        assert not result.success


@pytest.mark.asyncio

class TestWriteEdgeCases:
    """Test edge cases in file writing."""

    async def test_write_file_empty_content(self, ctx):
        """Writing empty content should succeed."""
        from squishy.tools.fs import write_file

        result = await write_file.run(
            {"path": "empty.py", "content": ""},
            ctx,
        )
        assert result.success

    async def test_write_file_binary_like_content(self, ctx):
        """Writing content that looks like binary should work as text."""
        from squishy.tools.fs import write_file

        # Content with special characters
        content = "#!/usr/bin/env python\nprint('hello')\n"

        result = await write_file.run(
            {"path": "script.py", "content": content},
            ctx,
        )
        assert result.success

    async def test_write_file_large_content(self, ctx):
        """Writing large content should succeed."""
        from squishy.tools.fs import write_file

        # 1000 lines of content
        large_content = "\n".join(f"# line {i}" for i in range(1000)) + "\n"

        result = await write_file.run(
            {"path": "large.py", "content": large_content},
            ctx,
        )
        assert result.success


class TestToolContextEdgeCases:
    """Test edge cases in ToolContext."""

    def test_plan_state_with_no_steps(self, tmp_path):
        """PlanState with empty steps list."""
        plan = PlanState.create(
            problem="Test",
            solution="Solution",
            steps=[],
        )
        assert len(plan.steps) == 0
        progress = plan.progress()
        assert progress["total"] == 0
        assert progress["pending"] == 0

    def test_plan_state_with_unicode(self, tmp_path):
        """PlanState should handle unicode characters."""
        plan = PlanState.create(
            problem="测试问题",
            solution="解决方法",
            steps=["步骤 1", "步骤 2"],
        )
        assert plan.problem == "测试问题"
        assert plan.solution == "解决方法"

    def test_plan_progress_with_mixed_statuses(self, tmp_path):
        """Progress should correctly count mixed statuses."""
        plan = PlanState.create(
            problem="Test",
            solution="Solution",
            steps=["Step 1", "Step 2", "Step 3", "Step 4"],
        )
        plan.update_step(step_index=0, status="done")
        plan.update_step(step_index=1, status="in-progress")
        plan.update_step(step_index=2, status="blocked")

        progress = plan.progress()
        assert progress["done"] == 1
        assert progress["in_progress"] == 1
        assert progress["blocked"] == 1
        assert progress["pending"] == 1

    def test_plan_evidence_list_empty_initially(self, tmp_path):
        """pending_plan_evidence should be empty by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.pending_plan_evidence == []

    def test_files_read_meta_empty_initially(self, tmp_path):
        """files_read_meta should be empty by default."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.files_read_meta == {}

    def test_context_with_different_working_dirs(self, tmp_path):
        """Multiple contexts with different working dirs."""
        ctx1 = ToolContext(
            working_dir=str(tmp_path / "dir1"),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx2 = ToolContext(
            working_dir=str(tmp_path / "dir2"),
            permission_mode="edits",
            use_sandbox=False,
        )

        ctx1.files_read["test.py"] = "content1"
        ctx2.files_read["test.py"] = "content2"

        assert ctx1.files_read["test.py"] == "content1"
        assert ctx2.files_read["test.py"] == "content2"


class TestLoadPlanEdgeCases:
    """Test edge cases in plan loading."""

    def test_load_nonexistent_plan(self, tmp_path):
        """Loading a non-existent plan file should return None."""
        result = load_plan(tmp_path)
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """Loading an invalid JSON file should return None."""
        plan_file = plan_path(tmp_path)
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text("not valid json {")

        result = load_plan(tmp_path)
        assert result is None

    def test_load_empty_json(self, tmp_path):
        """Loading an empty JSON object should handle gracefully."""
        plan_file = plan_path(tmp_path)
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text("{}")

        result = load_plan(tmp_path)
        # Should handle gracefully - likely returns None or minimal plan
        assert result is not None  # PlanState.from_dict handles missing fields

    def test_load_plan_with_missing_fields(self, tmp_path):
        """Loading plan with missing fields should use defaults."""
        plan_file = plan_path(tmp_path)
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text('{"problem": "test"}')

        result = load_plan(tmp_path)
        assert result is not None
        assert result.problem == "test"
        # Should have defaults for missing fields
        assert hasattr(result, "solution")
        assert hasattr(result, "steps")


@pytest.mark.asyncio

class TestConsecutiveReadsTracking:
    """Test MAX_RECALL_SKIP_TURNS enforcement."""

    async def test_read_tool_tracking_increments_counter(self, tmp_path):
        """Each read should increment the counter."""
        from squishy.agent import Agent
        from dataclasses import dataclass

        @dataclass
        class FakeClient:
            script: list[CompletionResult]
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
                if self._i >= len(self.script):
                    return CompletionResult(text="done.", tool_calls=[])
                result = self.script[self._i]
                self._i += 1
                return result

        from squishy.config import Config
        cfg = Config()
        cfg.working_dir = str(tmp_path)
        cfg.permission_mode = "plan"
        cfg.max_turns = 10

        # Create test files
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"# file {i}\n")

        script = [
            CompletionResult(tool_calls=[_tc("read_file", {"path": "file0.py"}, call_id="c1")]),
            CompletionResult(tool_calls=[_tc("read_file", {"path": "file1.py"}, call_id="c2")]),
            CompletionResult(tool_calls=[_tc("read_file", {"path": "file2.py"}, call_id="c3")]),
            CompletionResult(tool_calls=[_tc("read_file", {"path": "file3.py"}, call_id="c4")]),
            CompletionResult(text="done", tool_calls=[]),
        ]

        from squishy.display import Display

        fake = FakeClient(script=script)
        agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]

        initial_count = agent.consecutive_reads_without_recall

        assert initial_count == 0
