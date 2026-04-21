"""Tests for fmt_tokens, plan_task, update_plan, and plan display."""

from __future__ import annotations

import json

import pytest

from squishy.display import Display, Stats, fmt_tokens
from squishy.plan_state import load_plan, plan_path
from squishy.tools.base import ToolContext


class TestFmtTokens:
    def test_below_1k(self) -> None:
        assert fmt_tokens(500) == "500"

    def test_zero(self) -> None:
        assert fmt_tokens(0) == "0"

    def test_exactly_1k(self) -> None:
        assert fmt_tokens(1000) == "1.0K"

    def test_above_1k(self) -> None:
        assert fmt_tokens(1234) == "1.2K"

    def test_large(self) -> None:
        assert fmt_tokens(12345) == "12.3K"

    def test_with_context_window(self) -> None:
        result = fmt_tokens(1234, 8192)
        assert "1.2K" in result
        assert "(15%)" in result

    def test_with_context_window_zero_tokens(self) -> None:
        assert fmt_tokens(0, 8192) == "0 (0%)"

    def test_with_context_window_full(self) -> None:
        result = fmt_tokens(8192, 8192)
        assert "(100%)" in result

    def test_no_context_window(self) -> None:
        # No percentage when context_window is 0
        result = fmt_tokens(5000)
        assert "%" not in result
        assert "5.0K" in result


class TestStatsContextWindow:
    def test_default_context_window(self) -> None:
        s = Stats()
        assert s.context_window == 0

    def test_set_context_window(self) -> None:
        s = Stats()
        s.context_window = 8192
        assert s.context_window == 8192


@pytest.mark.asyncio
class TestPlanTask:
    async def test_plan_task_basic(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="plan", use_sandbox=False)
        result = await _plan_task(
            {
                "problem": "Tests are failing",
                "solution": "Fix the broken assertions",
                "steps": ["Read test file", "Fix assertion", "Run tests"],
                "files_to_modify": ["tests/test_foo.py"],
            },
            ctx,
        )
        assert result.success
        assert result.data["problem"] == "Tests are failing"
        assert result.data["solution"] == "Fix the broken assertions"
        assert len(result.data["steps"]) == 3
        assert all(s["status"] == "pending" for s in result.data["steps"])
        assert ctx.plan is not None
        assert plan_path(tmp_path).is_file()

    async def test_plan_task_missing_problem(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="plan", use_sandbox=False)
        result = await _plan_task(
            {
                "solution": "Fix it",
                "steps": ["step 1"],
            },
            ctx,
        )
        assert not result.success
        assert "problem" in result.error

    async def test_plan_task_missing_steps(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="plan", use_sandbox=False)
        result = await _plan_task(
            {
                "problem": "broken",
                "solution": "Fix it",
                "steps": [],
            },
            ctx,
        )
        assert not result.success
        assert "steps" in result.error

    async def test_plan_task_with_all_fields(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="plan", use_sandbox=False)
        result = await _plan_task(
            {
                "plan": "Fix failing tests",
                "problem": "Tests are failing",
                "solution": "Fix the broken assertions",
                "steps": ["Read file", "Fix it"],
                "files_to_create": ["new_file.py"],
                "files_to_modify": ["old_file.py"],
            },
            ctx,
        )
        assert result.success
        assert result.data["plan"] == "Fix failing tests"
        assert result.data["files_to_create"] == ["new_file.py"]
        assert result.data["files_to_modify"] == ["old_file.py"]
        persisted = load_plan(tmp_path)
        assert persisted is not None
        assert persisted.plan == "Fix failing tests"


@pytest.mark.asyncio
class TestUpdatePlan:
    async def test_update_plan_marks_done(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="edits", use_sandbox=False)
        await _plan_task(
            {
                "problem": "p",
                "solution": "s",
                "steps": ["step1", "step2", "step3"],
            },
            ctx,
        )
        result = await _update_plan({"step_index": 1, "status": "done"}, ctx)
        assert result.success
        assert result.data["progress"]["done"] == 1
        assert result.data["progress"]["pending"] == 2
        assert load_plan(tmp_path) is not None

    async def test_update_plan_no_active_plan(self, tmp_path) -> None:
        from squishy.tools.plan import _update_plan

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="edits", use_sandbox=False)
        result = await _update_plan({"step_index": 1, "status": "done"}, ctx)
        assert not result.success
        assert "no active plan" in result.error

    async def test_update_plan_out_of_range(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="edits", use_sandbox=False)
        await _plan_task(
            {"problem": "p", "solution": "s", "steps": ["step1"]},
            ctx,
        )
        result = await _update_plan({"step_index": 5, "status": "done"}, ctx)
        assert not result.success
        assert "out of range" in result.error

    async def test_update_plan_invalid_status(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="edits", use_sandbox=False)
        await _plan_task(
            {"problem": "p", "solution": "s", "steps": ["step1"]},
            ctx,
        )
        result = await _update_plan({"step_index": 1, "status": "invalid"}, ctx)
        assert not result.success

    async def test_update_plan_in_progress(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="edits", use_sandbox=False)
        await _plan_task(
            {"problem": "p", "solution": "s", "steps": ["step1", "step2"]},
            ctx,
        )
        result = await _update_plan({"step_index": 1, "status": "in-progress"}, ctx)
        assert result.success
        assert result.data["progress"]["in_progress"] == 1

    async def test_update_plan_attaches_evidence_and_blocked_note(self, tmp_path) -> None:
        from squishy.tools.plan import _plan_task, _update_plan

        ctx = ToolContext(working_dir=str(tmp_path), permission_mode="edits", use_sandbox=False)
        await _plan_task(
            {"problem": "p", "solution": "s", "steps": ["step1", "step2"]},
            ctx,
        )
        ctx.pending_plan_evidence.append({"kind": "edit_file", "path": "app.py", "detail": "patched bug"})
        result = await _update_plan(
            {"step_index": 1, "status": "blocked", "note": "waiting on user input"},
            ctx,
        )
        assert result.success
        assert result.data["note"] == "waiting on user input"
        assert result.data["evidence_count"] == 1
        persisted = json.loads(plan_path(tmp_path).read_text())
        assert persisted["steps"][0]["status"] == "blocked"
        assert persisted["steps"][0]["evidence"][0]["path"] == "app.py"


class TestPlanToolRestrictions:
    def test_plan_task_allowed_in_plan_mode(self) -> None:
        from squishy.tool_restrictions import get_allowed_tools

        allowed = get_allowed_tools("plan")
        assert "plan_task" in allowed
        assert "update_plan" in allowed

    def test_plan_task_allowed_in_all_modes(self) -> None:
        from squishy.tool_restrictions import get_allowed_tools

        for mode in ("plan", "edits", "yolo"):
            allowed = get_allowed_tools(mode)
            assert "plan_task" in allowed
            assert "update_plan" in allowed


class TestDisplayPlanPanel:
    def test_plan_panel_renders(self) -> None:
        """Smoke test that plan_panel doesn't crash."""
        display = Display()
        display.plan_panel({
            "plan": "Test plan",
            "problem": "Something broken",
            "solution": "Fix it",
            "steps": [
                {"description": "Step 1", "status": "done"},
                {"description": "Step 2", "status": "pending"},
            ],
            "files_to_create": ["new.py"],
            "files_to_modify": ["old.py"],
        })

    def test_plan_progress_renders(self) -> None:
        """Smoke test that plan_progress doesn't crash."""
        display = Display()
        display.plan_progress([
            {"description": "Step 1", "status": "done"},
            {"description": "Step 2", "status": "in-progress"},
            {"description": "Step 3", "status": "pending"},
        ])
