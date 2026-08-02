"""Tests for agent_safety: the nudge budget."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from squishy.agent_safety import can_nudge, record_nudge
from squishy.agent_state import LoopState


def _make_agent(
    messages: list[dict[str, Any]] | None = None,
    permission_mode: str = "bench",
    working_dir: str = "/workspace",
) -> MagicMock:
    agent = MagicMock()
    agent.messages = messages or []
    agent.config.permission_mode = permission_mode
    agent.config.max_system_nudges = 6
    agent.tool_ctx.working_dir = working_dir
    agent.display = None
    return agent


def _make_state(**kwargs) -> LoopState:
    return LoopState(start=time.monotonic(), **kwargs)


def _tc(name: str, args: dict | None = None) -> MagicMock:
    tc = MagicMock()
    tc.name = name
    tc.args = args or {}
    return tc


# -- can_nudge / record_nudge -------------------------------------------------

class TestNudgeGating:
    def test_can_nudge_with_gap(self):
        st = _make_state()
        st.last_nudge_turn = 0
        assert can_nudge(st, turn=3) is True

    def test_cannot_nudge_too_soon(self):
        st = _make_state()
        st.last_nudge_turn = 5
        assert can_nudge(st, turn=6) is False

    def test_cannot_nudge_at_cap(self):
        st = _make_state()
        st.total_nudges = 12
        assert can_nudge(st, turn=100) is False

    def test_record_nudge_updates_state(self):
        st = _make_state()
        record_nudge(st, turn=7)
        assert st.last_nudge_turn == 7
        assert st.total_nudges == 1



# -- inject_test_failure_nudge -----------------------------------------------

