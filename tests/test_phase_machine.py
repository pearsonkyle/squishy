"""Tests for phase_machine: phase transitions, tool sets, budgets."""
from __future__ import annotations

from unittest.mock import MagicMock

from squishy.phase_machine import (
    PhaseState,
    advance,
    check_transition,
    tools_for_phase,
)


def _tc(name: str, args: dict | None = None) -> MagicMock:
    tc = MagicMock()
    tc.name = name
    tc.args = args or {}
    return tc


# -- tools_for_phase ---------------------------------------------------------

class TestToolsForPhase:
    def test_explore_tools(self):
        tools = tools_for_phase("explore")
        assert "read_file" in tools
        assert "recall" in tools
        assert "run_command" in tools
        assert "edit_file" not in tools
        assert "plan_task" not in tools

    def test_plan_tools(self):
        tools = tools_for_phase("plan")
        assert "plan_task" in tools
        assert "save_note" in tools
        assert "recall" in tools
        assert "read_file" not in tools
        assert "edit_file" not in tools
        assert "run_command" not in tools

    def test_execute_tools(self):
        tools = tools_for_phase("execute")
        assert "edit_file" in tools
        assert "read_file" in tools
        assert "run_command" in tools
        assert "update_plan" in tools
        assert "plan_task" not in tools

    def test_verify_tools(self):
        tools = tools_for_phase("verify")
        assert "run_command" in tools
        assert "finish_plan" in tools
        assert "read_file" in tools
        assert "edit_file" not in tools

    def test_done_tools(self):
        tools = tools_for_phase("done")
        assert len(tools) == 0

    def test_unknown_phase_returns_empty(self):
        tools = tools_for_phase("nonexistent")
        assert len(tools) == 0


# -- explore phase -----------------------------------------------------------

class TestExplorePhase:
    def test_transition_after_recall(self):
        ps = PhaseState()
        dispatched = [(_tc("recall", {"query": "foo"}), {"success": True})]
        t = check_transition(ps, dispatched)
        assert t.new_phase == "plan"

    def test_transition_after_budget(self):
        ps = PhaseState(max_explore_turns=3)
        for i in range(3):
            dispatched = [(_tc("read_file", {"path": f"f{i}.py"}), {"success": True})]
            t = check_transition(ps, dispatched)
            if t.new_phase:
                break
        assert t.new_phase == "plan"
        assert ps.explore_turns == 3

    def test_no_transition_before_budget(self):
        ps = PhaseState(max_explore_turns=5)
        dispatched = [(_tc("read_file", {"path": "f.py"}), {"success": True})]
        t = check_transition(ps, dispatched)
        assert t.new_phase is None
        assert ps.explore_turns == 1

    def test_explore_turns_increment(self):
        ps = PhaseState(max_explore_turns=10)
        for _ in range(4):
            check_transition(ps, [(_tc("list_directory"), {"success": True})])
        assert ps.explore_turns == 4


# -- plan phase --------------------------------------------------------------

class TestPlanPhase:
    def test_transition_on_plan_success(self):
        ps = PhaseState(phase="plan")
        dispatched = [(_tc("plan_task", {"problem": "x"}), {"success": True})]
        t = check_transition(ps, dispatched)
        assert t.new_phase == "execute"

    def test_no_transition_on_plan_failure(self):
        ps = PhaseState(phase="plan")
        dispatched = [(_tc("plan_task", {"problem": "x"}), {"success": False})]
        t = check_transition(ps, dispatched)
        assert t.new_phase is None

    def test_transition_after_plan_budget(self):
        ps = PhaseState(phase="plan", max_plan_turns=2)
        check_transition(ps, [(_tc("save_note"), {"success": True})])
        t = check_transition(ps, [(_tc("save_note"), {"success": True})])
        assert t.new_phase == "execute"

    def test_plan_turns_increment(self):
        ps = PhaseState(phase="plan", max_plan_turns=5)
        check_transition(ps, [(_tc("save_note"), {"success": True})])
        check_transition(ps, [(_tc("save_note"), {"success": True})])
        assert ps.plan_turns == 2


# -- execute phase -----------------------------------------------------------

class TestExecutePhase:
    def test_transition_after_edit_and_test(self):
        ps = PhaseState(phase="execute")
        # Edit succeeds
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        assert ps.has_edit is True
        # Test run
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        assert t.new_phase == "verify"

    def test_no_transition_without_edit(self):
        ps = PhaseState(phase="execute")
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        assert t.new_phase is None

    def test_no_transition_without_test(self):
        ps = PhaseState(phase="execute")
        t = check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        assert t.new_phase is None

    def test_non_test_command_doesnt_trigger(self):
        ps = PhaseState(phase="execute")
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        t = check_transition(ps, [
            (_tc("run_command", {"command": "ls -la"}), {"success": True, "data": {"exit_code": 0}})
        ])
        assert t.new_phase is None

    def test_test_pass_skips_verify_to_done(self):
        """When test passes during execute, skip verify and go straight to done."""
        ps = PhaseState(phase="execute")
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        assert ps.has_edit is True
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": True, "data": {"exit_code": 0}})
        ])
        assert t.new_phase == "done"
        assert ps.test_passed_after_edit is True


# -- verify phase ------------------------------------------------------------

class TestVerifyPhase:
    def test_transition_to_done_on_test_pass(self):
        ps = PhaseState(phase="verify")
        ps.has_edit = True
        dispatched = [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": True, "data": {"exit_code": 0}})
        ]
        t = check_transition(ps, dispatched)
        assert t.new_phase == "done"

    def test_transition_to_done_on_finish_plan(self):
        ps = PhaseState(phase="verify")
        dispatched = [(_tc("finish_plan"), {"success": True})]
        t = check_transition(ps, dispatched)
        assert t.new_phase == "done"

    def test_transition_to_execute_on_test_fail(self):
        ps = PhaseState(phase="verify")
        dispatched = [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ]
        t = check_transition(ps, dispatched)
        assert t.new_phase == "execute"
        assert ps.fix_verify_cycles == 1

    def test_resets_edit_and_test_flags_on_fail_cycle(self):
        ps = PhaseState(phase="verify")
        ps.has_edit = True
        ps.has_test_run = True
        dispatched = [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ]
        check_transition(ps, dispatched)
        assert ps.has_edit is False
        assert ps.has_test_run is False
        assert ps.test_passed_after_edit is False  # must be reset on fail cycle

    def test_force_finish_at_cycle_limit(self):
        ps = PhaseState(phase="verify", max_fix_verify_cycles=3)
        ps.fix_verify_cycles = 2  # Will increment to 3
        ps.has_edit = True
        dispatched = [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ]
        t = check_transition(ps, dispatched)
        assert t.force_finish is True
        assert t.force_finish_success is True

    def test_force_finish_without_edits(self):
        ps = PhaseState(phase="verify", max_fix_verify_cycles=1)
        dispatched = [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ]
        t = check_transition(ps, dispatched)
        assert t.force_finish is True
        assert t.force_finish_success is False


# -- advance -----------------------------------------------------------------

class TestAdvance:
    def test_advance_updates_phase(self):
        ps = PhaseState(phase="explore")
        from squishy.phase_machine import Transition
        advance(ps, Transition(new_phase="plan"))
        assert ps.phase == "plan"

    def test_advance_noop_on_no_transition(self):
        ps = PhaseState(phase="execute")
        from squishy.phase_machine import Transition
        advance(ps, Transition())
        assert ps.phase == "execute"


# -- full lifecycle -----------------------------------------------------------

class TestFullLifecycle:
    def test_explore_to_done(self):
        """Walk through the full phase lifecycle."""
        ps = PhaseState(max_explore_turns=2, max_plan_turns=2, max_fix_verify_cycles=3)

        # explore: recall triggers transition
        t = check_transition(ps, [(_tc("recall", {"query": "bug"}), {"success": True})])
        assert t.new_phase == "plan"
        advance(ps, t)
        assert ps.phase == "plan"

        # plan: plan_task triggers transition
        t = check_transition(ps, [(_tc("plan_task", {"problem": "x"}), {"success": True})])
        assert t.new_phase == "execute"
        advance(ps, t)
        assert ps.phase == "execute"

        # execute: edit + test triggers transition
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        assert t.new_phase == "verify"
        advance(ps, t)
        assert ps.phase == "verify"

        # verify: test fails -> back to execute
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        assert t.new_phase == "execute"
        advance(ps, t)
        assert ps.phase == "execute"
        assert ps.fix_verify_cycles == 1

        # execute again: edit + test
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        advance(ps, t)
        assert ps.phase == "verify"

        # verify: test passes -> done
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": True, "data": {"exit_code": 0}})
        ])
        assert t.new_phase == "done"
        advance(ps, t)
        assert ps.phase == "done"

    def test_stale_test_passed_flag_does_not_cause_premature_done(self):
        """Regression: test_passed_after_edit must reset on verify→execute."""
        ps = PhaseState(phase="execute", max_fix_verify_cycles=5)

        # Cycle 1: edit, test fails → verify → execute
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        assert t.new_phase == "verify"
        advance(ps, t)

        # Verify: test fails again → back to execute
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        assert t.new_phase == "execute"
        advance(ps, t)
        assert ps.test_passed_after_edit is False

        # Cycle 2: edit, test passes → should go to done
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": True, "data": {"exit_code": 0}})
        ])
        assert t.new_phase == "done"  # via execute→done shortcut


class TestExecuteRunTestsNudge:
    def test_nudge_after_edit_without_test(self):
        """After edit + 2 turns without test run, should get a notification."""
        ps = PhaseState(phase="execute")
        # Turn 1: edit
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        assert ps.has_edit is True
        assert ps.execute_turns_since_edit == 1
        # Turn 2: update_plan (no test run)
        t = check_transition(ps, [(_tc("update_plan", {"step": 1}), {"success": True})])
        assert ps.execute_turns_since_edit == 2
        assert t.notification  # Should have a "run tests" nudge
        assert "run" in t.notification.lower() or "test" in t.notification.lower()
        assert t.new_phase is None  # No phase change, just a nudge

    def test_no_nudge_when_test_runs(self):
        """If a test command runs after edit, no nudge needed."""
        ps = PhaseState(phase="execute")
        check_transition(ps, [(_tc("edit_file", {"path": "f.py"}), {"success": True})])
        t = check_transition(ps, [
            (_tc("run_command", {"command": "pytest tests/"}), {"success": False, "data": {"exit_code": 1}})
        ])
        # Should transition to verify, not nudge
        assert t.new_phase == "verify"
