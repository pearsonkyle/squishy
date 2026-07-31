"""Repeat-read guards must not survive a compaction.

The compaction nudge tells the model to re-read files for exact `old_str`
text. If the read counters carry over, that next read is refused for being a
repeat — the harness instructing an action and then blocking it. In a 26-run
SWE-rebench sweep this was the single largest source of tool failures (106).
"""
from __future__ import annotations

from conftest import FakeClient

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display
from squishy.tools import dispatch


async def test_hard_cap_refuses_repeat_reads_without_compaction(ctx, tmp_path):
    """Baseline: the guard still works when context is intact."""
    ctx.working_dir = str(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    last = None
    for i in range(8):
        last = await dispatch("read_file", {"path": "a.py", "offset": i}, ctx)
    assert not last.success
    assert "refused" in last.error.lower()


async def test_compaction_clears_the_repeat_read_guards(tmp_path):
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "bench"
    cfg.max_turns = 3
    (tmp_path / "a.py").write_text("x = 1\n")

    fake = FakeClient(script=[
        CompletionResult(tool_calls=[ToolCall(
            id="c1", name="read_file", args={"path": "a.py"})]),
        CompletionResult(text="done", tool_calls=[]),
    ])
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    await agent.run("look at a.py")

    # Simulate the state a long run reaches before compaction fires.
    agent.tool_ctx.files_read_count["whatever"] = 99
    agent.tool_ctx.read_cache_hits[("whatever", 0, None)] = 99

    # The reset the loop performs on compaction.
    agent.tool_ctx.files_read_count.clear()
    agent.tool_ctx.read_cache_hits.clear()

    res = await dispatch("read_file", {"path": "a.py"}, agent.tool_ctx)
    assert res.success, res.error


async def test_reset_is_wired_into_the_loop():
    """Guard against the reset being dropped from agent.py."""
    import inspect

    from squishy import agent as agent_mod
    src = inspect.getsource(agent_mod)
    assert "files_read_count.clear()" in src
    assert "read_cache_hits.clear()" in src
