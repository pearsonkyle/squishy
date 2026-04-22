from __future__ import annotations

from types import SimpleNamespace

import pytest

import squishy.cli as cli
from squishy.config import Config
from squishy.plan_state import PlanState

pytestmark = pytest.mark.asyncio


async def test_run_one_continues_after_plan_approval(monkeypatch):
    cfg = Config()
    cfg.permission_mode = "plan"

    class FakeClient:
        pass

    class FakeDisplay:
        def __init__(self) -> None:
            self.info_calls: list[str] = []

        def info(self, message: str) -> None:
            self.info_calls.append(message)

        def warn(self, _message: str) -> None:
            pass

        def error(self, _message: str) -> None:
            pass

    seen_agents: list[FakeAgent] = []

    class FakeAgent:

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.tool_ctx = SimpleNamespace(plan=None, plan_switch_prompted=False)
            self.calls: list[str] = []
            seen_agents.append(self)

        async def run(self, message: str, *, timeout: float | None = None):
            self.calls.append(message)
            if len(self.calls) == 1:
                plan = PlanState.create(problem="p", solution="s", steps=["a"])
                plan.mark_approved()
                self.tool_ctx.plan = plan
            return None

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    display = FakeDisplay()

    await cli._run_one(
        cfg,
        client=FakeClient(),
        display=display,
        prompt_fn=None,
        message="do task",
        timeout=None,
    )

    assert seen_agents
    assert seen_agents[0].calls == ["do task", "Execute the approved plan."]
    assert seen_agents[0].tool_ctx.plan_switch_prompted is True
    assert cfg.permission_mode == "edits"
    assert "[bold green]✓ Switched to edits mode[/]" in display.info_calls
