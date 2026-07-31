from __future__ import annotations

from types import SimpleNamespace

import squishy.cli as cli
from squishy.config import Config
from squishy.plan_state import PlanState, has_plan_file, plan_path, save_plan


async def test_run_one_continues_after_plan_approval(monkeypatch):
    cfg = Config()
    cfg.permission_mode = "plan"

    class FakeClient:
        pass

    class FakeDisplay:
        def __init__(self) -> None:
            self.info_calls: list[str] = []
            self.mode: str = ""

        def info(self, message: str) -> None:
            self.info_calls.append(message)

        def warn(self, _message: str) -> None:
            pass

        def error(self, _message: str) -> None:
            pass

        def set_mode(self, mode: str) -> None:
            self.mode = mode

    seen_agents: list[FakeAgent] = []

    class FakeAgent:

        def __init__(self, *args, **kwargs) -> None:
            del args
            self.prompt_fn = kwargs.get("prompt_fn")
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

    async def _approve(*_a, **_k):
        return True

    # An interactive prompt_fn means a human approved the plan, so the
    # approve -> switch-to-edits -> execute flow is expected. (The
    # non-interactive case is covered by the escalation tests below.)
    await cli._run_one(
        cfg,
        client=FakeClient(),
        display=display,
        prompt_fn=_approve,
        message="do task",
        timeout=None,
    )

    assert seen_agents
    assert seen_agents[0].calls == ["do task", "Execute the approved plan."]
    assert seen_agents[0].tool_ctx.plan_switch_prompted is True
    assert cfg.permission_mode == "edits"
    assert "[bold green]✓ Switched to edits mode[/]" in display.info_calls


async def test_user_configured_model_via_args():
    args = SimpleNamespace(model="my-model")
    assert cli._user_configured_model(args) is True


async def test_user_configured_model_via_env(monkeypatch):
    monkeypatch.setenv("SQUISHY_MODEL", "env-model")
    args = SimpleNamespace(model=None)
    assert cli._user_configured_model(args) is True


async def test_user_configured_model_default(monkeypatch):
    monkeypatch.delenv("SQUISHY_MODEL", raising=False)
    args = SimpleNamespace(model=None)
    assert cli._user_configured_model(args) is False


async def test_run_one_clears_stale_plan(monkeypatch, tmp_path):
    """A one-shot invocation should not pick up a leftover plan from a
    previous interactive run."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"

    # Seed a stale plan on disk to simulate a leftover from a prior session.
    stale = PlanState.create(problem="old", solution="old", steps=["old"])
    save_plan(tmp_path, stale)
    assert has_plan_file(tmp_path)

    class FakeClient:
        pass

    class FakeDisplay:
        def info(self, _m: str) -> None: pass
        def warn(self, _m: str) -> None: pass
        def error(self, _m: str) -> None: pass
        def set_mode(self, _m: str) -> None: pass

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.tool_ctx = SimpleNamespace(plan=None, plan_switch_prompted=False)

        async def run(self, _message, *, timeout=None):
            return None

    monkeypatch.setattr(cli, "Agent", FakeAgent)

    await cli._run_one(
        cfg, client=FakeClient(), display=FakeDisplay(),
        prompt_fn=None, message="x", timeout=None,
    )

    # The stale plan file should have been cleared before the agent started.
    assert not plan_path(tmp_path).exists()


async def test_auto_execute_plan_skipped_without_human_approval(monkeypatch, tmp_path):
    """Non-interactive runs (pipe / non-TTY -m) auto-approve the plan, so
    auto-switching plan->edits would grant unreviewed write access."""
    from squishy import cli
    from squishy.config import Config
    from squishy.display import Display
    from squishy.plan_state import PlanState

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"

    class _Agent:
        def __init__(self):
            self.prompt_fn = None  # no human in the loop
            self.tool_ctx = type("C", (), {
                "plan": PlanState.create(problem="p", solution="s", steps=["a"]),
                "plan_switch_prompted": False,
            })()
            self.ran = False

        async def run(self, *_a, **_k):
            self.ran = True

    agent = _Agent()
    agent.tool_ctx.plan.mark_approved()
    await cli._auto_execute_plan(agent, cfg, Display(), None)

    assert cfg.permission_mode == "plan", "must not escalate to edits"
    assert agent.ran is False, "must not execute the plan unreviewed"


async def test_auto_execute_plan_runs_when_human_approved(tmp_path):
    """With an interactive prompt_fn the existing approve->execute UX stands."""
    from squishy import cli
    from squishy.config import Config
    from squishy.display import Display
    from squishy.plan_state import PlanState

    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "plan"

    async def _prompt(*_a, **_k):
        return True

    class _Agent:
        def __init__(self):
            self.prompt_fn = _prompt
            self.tool_ctx = type("C", (), {
                "plan": PlanState.create(problem="p", solution="s", steps=["a"]),
                "plan_switch_prompted": False,
            })()
            self.ran = False

        async def run(self, *_a, **_k):
            self.ran = True

    agent = _Agent()
    agent.tool_ctx.plan.mark_approved()
    await cli._auto_execute_plan(agent, cfg, Display(), None)

    assert cfg.permission_mode == "edits"
    assert agent.ran is True
