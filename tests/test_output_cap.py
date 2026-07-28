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


def test_unknown_window_keeps_configured_cap(tmp_path):
    assert _cap(0, 32_000, str(tmp_path)) == 32_000


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
