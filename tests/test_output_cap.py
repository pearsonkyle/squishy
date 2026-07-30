"""Window-aware tool-output cap: a single result can't blow a small window."""

from __future__ import annotations

from squishy.agent import Agent
from squishy.config import Config
from squishy.tools.base import ToolResult


class _Client:
    def __init__(self, context_window: int) -> None:
        self.context_window = context_window


def _cap(context_window: int, configured: int = 32_000, tmp="/tmp") -> int:
    cfg = Config()
    cfg.working_dir = tmp
    cfg.max_tool_output_chars = configured
    agent = Agent(cfg, _Client(context_window), display=None)  # type: ignore[arg-type]
    return agent.tool_ctx.max_tool_output_chars


def test_unknown_window_uses_assumed_window(tmp_path):
    """An endpoint that doesn't advertise context_length still gets a bounded
    cap, derived from assumed_context_window (32768*3.5/8 = 14336) rather than
    leaving results effectively uncapped."""
    assert _cap(0, 32_000, str(tmp_path)) == 14_336


def test_configured_cap_still_bounds_a_large_assumed_window(tmp_path):
    assert _cap(0, 5_000, str(tmp_path)) == 5_000


def test_small_window_reduces_cap(tmp_path):
    # 8k-token window -> ~1/8 in chars = 3500 -> floored to 4000.
    assert _cap(8192, 32_000, str(tmp_path)) == 4000
    # 16k window -> 16384*3.5/8 = 7168.
    assert _cap(16384, 32_000, str(tmp_path)) == 7168


def test_large_window_never_exceeds_configured(tmp_path):
    # 128k window would compute a huge cap; stays at the configured value.
    assert _cap(131072, 32_000, str(tmp_path)) == 32_000


def test_to_message_respects_limit():
    big = {"content": "x" * 5000}
    short = ToolResult(True, data=big).to_message(1000)
    assert len(short) < 2000
    assert "snipped" in short
    # Default limit leaves small payloads intact.
    small = ToolResult(True, data={"content": "hi"}).to_message()
    assert "hi" in small


# --- effective context window (live-testing finding) ----------------------

def _agent(tmp, *, reported=0, override=0, assumed=32_768):
    cfg = Config()
    cfg.working_dir = str(tmp)
    cfg.context_window = override
    cfg.assumed_context_window = assumed
    return Agent(cfg, _Client(reported), display=None)  # type: ignore[arg-type]


def test_context_window_falls_back_to_assumed(tmp_path):
    """Endpoints that don't advertise context_length (LM Studio, llama.cpp)
    must still get compaction + history sizing, not silently disabled."""
    a = _agent(tmp_path, reported=0)
    assert a._context_window() == 32_768


def test_context_window_prefers_endpoint_value(tmp_path):
    a = _agent(tmp_path, reported=8192)
    assert a._context_window() == 8192


def test_context_window_explicit_override_wins(tmp_path):
    a = _agent(tmp_path, reported=8192, override=65536)
    assert a._context_window() == 65536


def test_output_cap_uses_assumed_window(tmp_path):
    """With no advertised window the cap now derives from the assumed one."""
    a = _agent(tmp_path, reported=0, assumed=8192)
    assert a.tool_ctx.max_tool_output_chars == 4000
