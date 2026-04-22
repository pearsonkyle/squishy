from __future__ import annotations

from types import SimpleNamespace

import pytest

import squishy.cli as cli
from squishy.config import Config

pytestmark = pytest.mark.asyncio


async def test_run_one_continues_after_plan_approval(monkeypatch):
    cfg = Config()
    cfg.permission_mode = "plan"

    class FakeDisplay:
        def __init__(self) -> None:
            self.info_calls: list[str] = []

        def info(self, message: str) -> None:
            self.info_calls.append(message)

        def warn(self, _message: str) -> None:
            pass

        def error(self, _message: str) -> None:
            pass

    class FakeAgent:
        last: FakeAgent | None = None

        def __init__(self, *_args, **_kwargs) -> None:
            self.tool_ctx = SimpleNamespace(plan=None, plan_switch_prompted=False)
            self.calls: list[str] = []
            FakeAgent.last = self

        async def run(self, message: str, *, timeout=None):  # type: ignore[no-untyped-def]
            self.calls.append(message)
            if len(self.calls) == 1:
                self.tool_ctx.plan = SimpleNamespace(approved=True)
            return None

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    display = FakeDisplay()

    await cli._run_one(cfg, client=object(), display=display, prompt_fn=None, message="do task", timeout=None)

    assert FakeAgent.last is not None
    assert FakeAgent.last.calls == ["do task", "Execute the approved plan."]
    assert cfg.permission_mode == "edits"
    assert "[bold green]✓ Switched to edits mode[/]" in display.info_calls
