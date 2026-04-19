"""Tests for exit_plan_mode and update_plan_step tools."""

from __future__ import annotations

import pytest

from squishy.config import Config
from squishy.display import Display
from squishy.plan_tracker import PlanTracker
from squishy.tool_restrictions import check_permission, get_allowed_tools
from squishy.tools import dispatch
from squishy.tools.base import ToolContext
from squishy.tools.plan import exit_plan_mode, update_plan_step


def _make_ctx(mode: str = "plan", *, approve: bool | None = True) -> tuple[ToolContext, Config, PlanTracker, Display]:
    cfg = Config()
    cfg.permission_mode = mode
    display = Display()
    tracker = display.tracker

    async def approve_fn(question: str) -> bool:
        assert approve is not None, "approve_fn called but approve=None"
        return approve

    ctx = ToolContext(
        working_dir="/tmp",
        permission_mode=mode,
        cfg_ref=cfg,
        tracker=tracker,
        display=display,
        approve_fn=approve_fn if approve is not None else None,
    )
    return ctx, cfg, tracker, display


@pytest.mark.asyncio
async def test_exit_plan_mode_requires_plan_and_steps() -> None:
    ctx, cfg, tracker, _ = _make_ctx()
    result = await exit_plan_mode.run({"plan": ""}, ctx)
    assert not result.success
    assert "plan" in result.error

    result = await exit_plan_mode.run({"plan": "do the thing"}, ctx)
    assert not result.success
    assert "implementation_steps" in result.error


@pytest.mark.asyncio
async def test_exit_plan_mode_approved_flips_mode_and_populates_tracker() -> None:
    ctx, cfg, tracker, _ = _make_ctx(mode="plan", approve=True)

    result = await exit_plan_mode.run(
        {
            "plan": "refactor x to y",
            "problem": "x is wrong",
            "solution_steps": ["read x", "rewrite x"],
            "files_create": ["new.py"],
            "files_modify": ["old.py"],
            "implementation_steps": ["step one", "step two", "step three"],
        },
        ctx,
    )

    assert result.success
    assert result.data["approved"] is True
    assert cfg.permission_mode == "edits"
    assert tracker.approved
    assert len(tracker.steps) == 3
    assert tracker.steps[0].description == "step one"
    assert tracker.steps[0].status == "pending"
    assert tracker.plan == "refactor x to y"


@pytest.mark.asyncio
async def test_exit_plan_mode_declined_stays_in_plan_mode() -> None:
    ctx, cfg, tracker, _ = _make_ctx(mode="plan", approve=False)

    result = await exit_plan_mode.run(
        {
            "plan": "refactor x to y",
            "implementation_steps": ["one", "two"],
        },
        ctx,
    )

    assert result.success
    assert result.data["approved"] is False
    assert "refine" in result.data["note"]
    assert cfg.permission_mode == "plan"
    assert not tracker.approved
    assert tracker.steps == []


@pytest.mark.asyncio
async def test_exit_plan_mode_no_approve_fn_refuses() -> None:
    ctx, _, _, _ = _make_ctx(mode="plan", approve=None)
    result = await exit_plan_mode.run(
        {"plan": "x", "implementation_steps": ["one"]},
        ctx,
    )
    assert not result.success
    assert "interactive" in result.error


@pytest.mark.asyncio
async def test_update_plan_step_marks_progress() -> None:
    ctx, cfg, tracker, _ = _make_ctx(mode="plan", approve=True)
    await exit_plan_mode.run(
        {"plan": "p", "implementation_steps": ["a", "b", "c"]},
        ctx,
    )

    result = await update_plan_step.run({"index": 0, "status": "in_progress"}, ctx)
    assert result.success
    assert tracker.steps[0].status == "in_progress"

    result = await update_plan_step.run({"index": 0, "status": "done"}, ctx)
    assert result.success
    assert tracker.steps[0].status == "done"

    result = await update_plan_step.run({"index": 99, "status": "done"}, ctx)
    assert not result.success
    assert "no step with index 99" in result.error

    result = await update_plan_step.run({"index": 1, "status": "bogus"}, ctx)
    assert not result.success
    assert "invalid status" in result.error


@pytest.mark.asyncio
async def test_update_plan_step_rejects_when_no_active_plan() -> None:
    ctx, _, _, _ = _make_ctx(mode="edits", approve=True)
    result = await update_plan_step.run({"index": 0, "status": "done"}, ctx)
    assert not result.success
    assert "no active plan" in result.error


def test_exit_plan_mode_only_allowed_in_plan_mode() -> None:
    allowed, reason = check_permission("exit_plan_mode", "plan")
    assert allowed

    allowed, reason = check_permission("exit_plan_mode", "edits")
    assert not allowed
    assert "plan mode" in reason

    allowed, reason = check_permission("exit_plan_mode", "yolo")
    # yolo allows everything — we honor the blanket permission.
    assert allowed

    # update_plan_step is available in every mode so progress can be tracked
    # after the session has flipped to edits.
    for mode in ("plan", "edits", "yolo"):
        assert check_permission("update_plan_step", mode)[0]


def test_get_allowed_tools_includes_plan_tools() -> None:
    assert "exit_plan_mode" in get_allowed_tools("plan")
    assert "exit_plan_mode" not in get_allowed_tools("edits")
    assert "update_plan_step" in get_allowed_tools("plan")
    assert "update_plan_step" in get_allowed_tools("edits")


@pytest.mark.asyncio
async def test_dispatch_routes_plan_tool() -> None:
    ctx, cfg, tracker, _ = _make_ctx(mode="plan", approve=True)
    outcome = await dispatch(
        "exit_plan_mode",
        {"plan": "p", "implementation_steps": ["x"]},
        ctx,
    )
    assert outcome.success
    assert tracker.approved
