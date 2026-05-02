"""Tests for agent_phases: cache_problem_text, maybe_reanchor_problem, update_phase."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from squishy.agent_phases import cache_problem_text, maybe_reanchor_problem, update_phase
from squishy.agent_state import LoopState


def _make_agent(messages: list[dict[str, Any]], permission_mode: str = "bench") -> MagicMock:
    """Build a minimal mock Agent with .messages and .config."""
    agent = MagicMock()
    agent.messages = messages
    agent.config.permission_mode = permission_mode
    agent.config.max_explore_turns = 3
    agent.config.max_fix_verify_cycles = 6
    agent.config.max_post_edit_read_turns = 4
    agent.config.max_system_nudges = 6
    agent.display = None
    return agent


def _make_state(**kwargs) -> LoopState:
    return LoopState(start=time.monotonic(), **kwargs)


# -- cache_problem_text --------------------------------------------------------

class TestCacheProblemText:
    def test_extracts_problem_text(self):
        messages = [
            {"role": "user", "content": "## Problem\nSomething is broken in foo.py\n\n## Steps"},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.problem_text == "Something is broken in foo.py\n\n## Steps"

    def test_truncates_long_problem_text(self):
        long_text = "x" * 2000
        messages = [
            {"role": "user", "content": f"## Problem\n{long_text}"},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert len(st.problem_text) == 1503  # 1500 + "..."
        assert st.problem_text.endswith("...")

    def test_extracts_fail_to_pass_tests(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "## Problem\nBug.\n\n"
                    "## Failing Tests\n"
                    "- `tests/test_foo.py::TestBar::test_baz`\n"
                    "- `tests/test_foo.py::test_qux`\n"
                    "\n## Other\nStuff"
                ),
            },
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.fail_to_pass_tests == [
            "tests/test_foo.py::TestBar::test_baz",
            "tests/test_foo.py::test_qux",
        ]

    def test_no_failing_tests_section(self):
        messages = [
            {"role": "user", "content": "## Problem\nBug."},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.fail_to_pass_tests == []

    def test_skips_system_messages(self):
        messages = [
            {"role": "user", "content": "[system] not a problem statement"},
            {"role": "user", "content": "## Problem\nReal bug."},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.problem_text == "Real bug."

    def test_no_problem_section(self):
        messages = [
            {"role": "user", "content": "Just fix it."},
        ]
        agent = _make_agent(messages)
        st = _make_state()
        cache_problem_text(agent, st)
        assert st.problem_text is None


# -- maybe_reanchor_problem ----------------------------------------------------

class TestMaybeReanchorProblem:
    def test_injects_problem_text(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.problem_text = "The bug is X."
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 1
        assert "The bug is X." in messages[0]["content"]
        assert st.last_reanchor_turn == 10

    def test_includes_fail_to_pass_tests(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.problem_text = "Bug."
        st.fail_to_pass_tests = ["tests/test_a.py::test_1", "tests/test_b.py::test_2"]
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        content = messages[0]["content"]
        assert "`tests/test_a.py::test_1`" in content
        assert "`tests/test_b.py::test_2`" in content

    def test_skips_when_too_recent(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.problem_text = "Bug."
        st.last_reanchor_turn = 8

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 0

    def test_skips_non_bench_mode(self):
        messages: list[dict] = []
        agent = _make_agent(messages, permission_mode="edits")
        st = _make_state()
        st.problem_text = "Bug."
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 0

    def test_skips_when_no_problem_text(self):
        messages: list[dict] = []
        agent = _make_agent(messages)
        st = _make_state()
        st.last_reanchor_turn = 0

        maybe_reanchor_problem(agent, st, turn=10)
        assert len(messages) == 0


# -- update_phase --------------------------------------------------------------

class TestUpdatePhase:
    def _tc(self, name: str, args: dict | None = None) -> MagicMock:
        tc = MagicMock()
        tc.name = name
        tc.args = args or {}
        return tc

    def test_edit_transitions_to_fix(self):
        agent = _make_agent([])
        st = _make_state()
        assert st.phase == "explore"
        tc = self._tc("edit_file")
        update_phase(agent, st, [(tc, {"success": True})], turn=1)
        assert st.phase == "fix"

    def test_test_after_fix_transitions_to_verify(self):
        agent = _make_agent([])
        st = _make_state(phase="fix")
        tc = self._tc("run_command", {"command": "pytest tests/"})
        update_phase(agent, st, [(tc, {"success": True})], turn=2)
        assert st.phase == "verify"
        assert st.fix_verify_cycles == 1

    def test_verify_transitions_back_to_fix(self):
        agent = _make_agent([])
        st = _make_state(phase="verify")
        tc = self._tc("read_file", {"path": "foo.py"})
        update_phase(agent, st, [(tc, {"success": True})], turn=3)
        assert st.phase == "fix"

    def test_non_bench_returns_none(self):
        agent = _make_agent([], permission_mode="edits")
        st = _make_state()
        result = update_phase(agent, st, [], turn=1)
        assert result is None

    def test_fix_verify_budget_exhausted(self):
        from squishy.agent_state import TaskResult as TR
        agent = _make_agent([])
        # Mock _build_result to return a real TaskResult.
        agent._build_result = MagicMock(return_value=TR(success=True, final_text="done"))
        st = _make_state(phase="fix")
        st.fix_verify_cycles = 6
        st.files_edited.add("foo.py")
        st.test_passed_after_edit = True
        result = update_phase(agent, st, [], turn=10)
        assert result is not None
        assert result.success is True
