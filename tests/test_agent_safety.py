"""Tests for agent_safety: goal drift, edit failure tracking, nudge turn tracking."""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

from squishy.agent_safety import (
    can_nudge,
    check_goal_drift,
    inject_test_failure_nudge,
    record_nudge,
    track_edit_failure,
)
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


# -- check_goal_drift ---------------------------------------------------------

class TestCheckGoalDrift:
    def test_nudges_on_env_errors(self):
        agent = _make_agent()
        st = _make_state()
        st.problem_files = {"src/core.py"}
        st.env_error_count = 2

        # Two env_fix_files + 2 env_errors = should nudge
        st.env_fix_files = {"setup.py", "requirements.txt"}

        tc = _tc("run_command", {"command": "pytest"})
        outcome = {"success": False, "data": {"stderr": "ImportError: No module named xyz"}}
        check_goal_drift(agent, st, tc, outcome, turn=5)

        # env_error_count incremented to 3, then should_nudge fires
        assert len(agent.messages) >= 1
        assert "environment" in agent.messages[-1]["content"].lower()

    def test_clears_env_fix_files_after_nudge(self):
        agent = _make_agent()
        st = _make_state()
        st.env_fix_files = {"setup.py", "requirements.txt"}
        st.env_error_count = 2

        tc = _tc("run_command", {"command": "pytest"})
        outcome = {"success": False, "data": {"stderr": "ImportError: No module"}}
        check_goal_drift(agent, st, tc, outcome, turn=5)

        # After nudge, env_fix_files should be cleared
        assert len(st.env_fix_files) == 0
        assert st.env_error_count == 0

    def test_passes_correct_turn(self):
        agent = _make_agent()
        st = _make_state()
        st.env_error_count = 3  # Will trigger nudge

        tc = _tc("run_command", {"command": "pytest"})
        outcome = {"success": False, "data": {"stderr": "ImportError"}}
        check_goal_drift(agent, st, tc, outcome, turn=15)

        # last_nudge_turn should be 15, not 0
        assert st.last_nudge_turn == 15

    def test_skips_non_bench_mode(self):
        agent = _make_agent(permission_mode="edits")
        st = _make_state()
        st.env_error_count = 10

        tc = _tc("run_command", {"command": "pytest"})
        outcome = {"success": False, "data": {"stderr": "ImportError"}}
        check_goal_drift(agent, st, tc, outcome, turn=5)
        assert len(agent.messages) == 0


# -- track_edit_failure --------------------------------------------------------

class TestTrackEditFailure:
    def test_nudge_at_3_failures(self):
        agent = _make_agent()
        st = _make_state()

        tc = _tc("edit_file", {"path": "foo.py"})
        outcome = {"success": False}

        for i in range(3):
            track_edit_failure(agent, st, tc, outcome, turn=i + 1)

        assert st.edit_failures_per_file["foo.py"] == 3
        assert st.total_edit_failures == 3
        assert any("failed edits" in m["content"].lower() for m in agent.messages)

    def test_guidance_at_5_failures(self):
        """At 5+ failures, guidance should still appear but not be CRITICAL."""
        agent = _make_agent()
        st = _make_state()

        tc = _tc("edit_file", {"path": "foo.py"})
        outcome = {"success": False}

        for i in range(5):
            track_edit_failure(agent, st, tc, outcome, turn=i + 1)

        # Should have guidance messages but no CRITICAL
        assert any("old_str" in m["content"] for m in agent.messages)
        assert not any("CRITICAL" in m["content"] for m in agent.messages)

    def test_passes_correct_turn(self):
        agent = _make_agent()
        st = _make_state()

        tc = _tc("edit_file", {"path": "foo.py"})
        outcome = {"success": False}

        for i in range(3):
            track_edit_failure(agent, st, tc, outcome, turn=10 + i)

        # last_nudge_turn should be around 12, not 0
        assert st.last_nudge_turn >= 10

    def test_resets_on_success(self):
        agent = _make_agent()
        st = _make_state()

        tc = _tc("edit_file", {"path": "foo.py"})
        track_edit_failure(agent, st, tc, {"success": False}, turn=1)
        track_edit_failure(agent, st, tc, {"success": False}, turn=2)
        track_edit_failure(agent, st, tc, {"success": True}, turn=3)
        assert st.edit_failures_per_file["foo.py"] == 0

    def test_skips_non_edit(self):
        agent = _make_agent()
        st = _make_state()

        tc = _tc("read_file", {"path": "foo.py"})
        track_edit_failure(agent, st, tc, {"success": False}, turn=1)
        assert st.total_edit_failures == 0

    def test_v6d_escape_hatch_fires_at_count_2_for_old_str_not_found(self):
        """v6d: 2 consecutive `old_str not found` errors trigger a
        targeted nudge that mandates a re-read, one turn earlier than
        the generic 3-failure message."""
        agent = _make_agent()
        st = _make_state()

        tc = _tc("edit_file", {"path": "foo/bar.py", "old_str": "x"})
        outcome = {"success": False, "error": "old_str not found in file."}

        # First failure: no nudge yet (count 1, threshold is 2).
        track_edit_failure(agent, st, tc, outcome, turn=1)
        assert len(agent.messages) == 0

        # Second failure: escape-hatch nudge fires.
        track_edit_failure(agent, st, tc, outcome, turn=3)
        assert len(agent.messages) == 1
        msg = agent.messages[0]["content"]
        assert "stop guessing" in msg.lower()
        assert "foo/bar.py" in msg
        assert 'read_file(path="foo/bar.py")' in msg

    def test_v6d_escape_hatch_skips_other_errors(self):
        """v6d: only `old_str not found` triggers the escape hatch;
        other failure modes (permission errors, etc.) still rely on
        the generic 3-failure nudge."""
        agent = _make_agent()
        st = _make_state()

        tc = _tc("edit_file", {"path": "foo.py", "old_str": "x"})
        outcome = {"success": False, "error": "permission denied"}

        track_edit_failure(agent, st, tc, outcome, turn=1)
        track_edit_failure(agent, st, tc, outcome, turn=3)
        # No escape-hatch nudge — only permission-style failures so far.
        assert not any("stop guessing" in m["content"].lower() for m in agent.messages)


# -- inject_test_failure_nudge -----------------------------------------------

class TestTestFailureNudge:
    def _outcome(self, exit_code=1, test_summary=None):
        data = {"exit_code": exit_code, "stderr": "", "stdout": ""}
        if test_summary is not None:
            data["test_summary"] = test_summary
        return {"success": exit_code == 0, "data": data}

    def test_fires_on_first_failure(self):
        """Nudge fires even on the very first test failure (cycle 0)."""
        agent = _make_agent()
        st = _make_state()
        st.fix_verify_cycles = 0

        tc = _tc("run_command", {"command": "pytest tests/"})
        outcome = self._outcome(
            exit_code=1,
            test_summary={
                "passed": 5, "failed": 2, "errors": 0,
                "failures": [
                    {"test": "tests/test_foo.py::test_bar", "error": "AssertionError: 1 != 2"},
                    {"test": "tests/test_foo.py::test_baz", "error": "TypeError: bad"},
                ],
            },
        )
        inject_test_failure_nudge(agent, st, tc, outcome, turn=5)

        assert len(agent.messages) >= 1
        content = agent.messages[-1]["content"]
        assert "5 passed" in content
        assert "2 failed" in content
        assert "test_bar" in content
        assert "AssertionError" in content

    def test_shows_progress_when_failures_decrease(self):
        agent = _make_agent()
        st = _make_state()
        st.last_test_failure_count = 5
        st.last_test_failures = ["a", "b", "c", "d", "e"]

        tc = _tc("run_command", {"command": "pytest tests/"})
        outcome = self._outcome(
            exit_code=1,
            test_summary={
                "passed": 8, "failed": 2, "errors": 0,
                "failures": [
                    {"test": "a", "error": "err"},
                    {"test": "b", "error": "err"},
                ],
            },
        )
        inject_test_failure_nudge(agent, st, tc, outcome, turn=10)

        content = agent.messages[-1]["content"]
        assert "decreased from 5 to 2" in content
        assert st.last_test_failure_count == 2

    def test_warns_when_failures_increase(self):
        agent = _make_agent()
        st = _make_state()
        st.last_test_failure_count = 2
        st.last_test_failures = ["a", "b"]

        tc = _tc("run_command", {"command": "pytest tests/"})
        outcome = self._outcome(
            exit_code=1,
            test_summary={
                "passed": 3, "failed": 5, "errors": 0,
                "failures": [{"test": f"t{i}", "error": "err"} for i in range(5)],
            },
        )
        inject_test_failure_nudge(agent, st, tc, outcome, turn=10)

        content = agent.messages[-1]["content"]
        assert "INCREASED failures from 2 to 5" in content

    def test_same_failures_suggest_different_approach(self):
        agent = _make_agent()
        st = _make_state()
        st.last_test_failure_count = 2
        st.last_test_failures = ["tests/a.py::test_x", "tests/a.py::test_y"]

        tc = _tc("run_command", {"command": "pytest tests/"})
        outcome = self._outcome(
            exit_code=1,
            test_summary={
                "passed": 5, "failed": 2, "errors": 0,
                "failures": [
                    {"test": "tests/a.py::test_x", "error": "err"},
                    {"test": "tests/a.py::test_y", "error": "err"},
                ],
            },
        )
        inject_test_failure_nudge(agent, st, tc, outcome, turn=10)

        content = agent.messages[-1]["content"]
        assert "no effect" in content.lower() or "DIFFERENT" in content

    def test_no_critical_at_4_cycles(self):
        """At 4+ cycles, structured feedback fires but no CRITICAL escalation."""
        agent = _make_agent()
        st = _make_state()
        st.fix_verify_cycles = 4

        tc = _tc("run_command", {"command": "pytest tests/"})
        outcome = self._outcome(
            exit_code=1,
            test_summary={
                "passed": 3, "failed": 1, "errors": 0,
                "failures": [{"test": "t", "error": "err"}],
            },
        )
        inject_test_failure_nudge(agent, st, tc, outcome, turn=20)

        content = agent.messages[-1]["content"]
        # Should have structured feedback but NOT CRITICAL escalation
        assert "1 failed" in content
        assert "CRITICAL" not in content

    def test_skips_non_test_commands(self):
        agent = _make_agent()
        st = _make_state()

        tc = _tc("run_command", {"command": "ls -la"})
        outcome = self._outcome(exit_code=1)
        inject_test_failure_nudge(agent, st, tc, outcome, turn=5)
        assert len(agent.messages) == 0

    def test_skips_passing_tests(self):
        agent = _make_agent()
        st = _make_state()

        tc = _tc("run_command", {"command": "pytest tests/"})
        outcome = self._outcome(exit_code=0)
        inject_test_failure_nudge(agent, st, tc, outcome, turn=5)
        assert len(agent.messages) == 0
