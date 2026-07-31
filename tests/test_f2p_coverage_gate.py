"""F5: F2P file-coverage gate before ``finish_plan``.

When ``FAIL_TO_PASS`` spans multiple test files (e.g. scico-561's
``test_xray_2d.py::test_matched_adjoint`` AND
``test_xray_3d.py::test_matched_adjoint``), the agent must exercise
*every* distinct F2P file before ``test_passed_after_edit`` flips True.
Otherwise the agent fixes one class, runs only the 2D test, sees pass,
and ships without ever touching the 3D class — exactly what v27.1 did.

After one intercept the gate releases (single-shot, like A5) so a
genuinely degraded test environment cannot trap the agent.
"""
from __future__ import annotations

from squishy.agent_state import (
    distinct_f2p_files,
    f2p_files_in_command,
)
from squishy.agent_state import (
    test_covers_fail_to_pass as _covers,
)
from squishy.client import ToolCall
from squishy.phase_machine import (
    PhaseState,
    check_finish_plan_gate,
    check_transition,
)

F2P = [
    "scico/test/linop/xray/test_xray_2d.py::test_matched_adjoint",
    "scico/test/linop/xray/test_xray_3d.py::test_matched_adjoint",
]


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


# ── distinct_f2p_files / f2p_files_in_command ────────────────────────


def test_distinct_f2p_files_dedupes_by_file():
    out = distinct_f2p_files([
        "tests/a.py::test_one",
        "tests/a.py::test_two[x]",
        "tests/b.py::TestC::test_three",
    ])
    assert out == {"tests/a.py", "tests/b.py"}


def test_f2p_files_in_command_matches_explicit_path():
    cmd = "pytest scico/test/linop/xray/test_xray_2d.py -x"
    covered = f2p_files_in_command(cmd, F2P)
    assert covered == {"scico/test/linop/xray/test_xray_2d.py"}


def test_f2p_files_in_command_bare_pytest_is_full_suite():
    # No path arg → conservatively treat as covering every F2P file.
    covered = f2p_files_in_command("pytest -x", F2P)
    assert covered == set(distinct_f2p_files(F2P))


def test_f2p_files_in_command_unrelated_test_covers_nothing():
    cmd = "pytest tests/unrelated.py"
    assert f2p_files_in_command(cmd, F2P) == set()


def test_covers_helper_is_now_file_aware():
    # Backwards-compat: bool-returning helper should still True-out for
    # any matching file, but False when the command runs nothing related.
    assert _covers(
        "pytest scico/test/linop/xray/test_xray_2d.py", F2P,
    )
    assert not _covers(
        "pytest tests/unrelated.py", F2P,
    )
    # No F2P → fall back to True.
    assert _covers("pytest", [])


# ── check_transition: F5 gating ──────────────────────────────────────


def _ps_with_edit() -> PhaseState:
    """Phase state that already has an edit and is in execute phase."""
    return PhaseState(phase="execute", has_edit=True, fail_to_pass=list(F2P))


def test_transition_does_not_pass_after_only_one_f2p_file():
    """Running only the 2D test must NOT flip test_passed_after_edit."""
    ps = _ps_with_edit()
    dispatched = [(
        _tc("run_command", {"command": "pytest scico/test/linop/xray/test_xray_2d.py"}),
        {"success": True, "data": {"exit_code": 0}},
    )]
    check_transition(ps, dispatched)
    assert ps.f2p_files_covered == {"scico/test/linop/xray/test_xray_2d.py"}
    assert ps.test_passed_after_edit is False, (
        "F5: only 1/2 F2P files covered, must not declare pass"
    )


def test_transition_passes_when_all_f2p_files_covered():
    """Running both files (across two turns) eventually flips the flag."""
    ps = _ps_with_edit()
    check_transition(ps, [(
        _tc("run_command", {"command": "pytest scico/test/linop/xray/test_xray_2d.py"}),
        {"success": True, "data": {"exit_code": 0}},
    )])
    assert ps.test_passed_after_edit is False
    check_transition(ps, [(
        _tc("run_command", {"command": "pytest scico/test/linop/xray/test_xray_3d.py"}),
        {"success": True, "data": {"exit_code": 0}},
    )])
    assert ps.f2p_files_covered == set(distinct_f2p_files(F2P))
    assert ps.test_passed_after_edit is True


def test_transition_passes_for_bare_pytest_full_suite():
    """A bare ``pytest`` command is treated as covering everything."""
    ps = _ps_with_edit()
    check_transition(ps, [(
        _tc("run_command", {"command": "pytest -x"}),
        {"success": True, "data": {"exit_code": 0}},
    )])
    assert ps.test_passed_after_edit is True


def test_transition_no_f2p_keeps_legacy_behavior():
    """Without F2P (interactive / terminal-bench), exit-0 test → pass."""
    ps = PhaseState(phase="execute", has_edit=True, fail_to_pass=[])
    check_transition(ps, [(
        _tc("run_command", {"command": "pytest tests/anything.py"}),
        {"success": True, "data": {"exit_code": 0}},
    )])
    assert ps.test_passed_after_edit is True


def test_transition_test_summary_failures_block_coverage():
    """A false-positive exit 0 with failures in test_summary must not
    grant coverage credit."""
    ps = _ps_with_edit()
    check_transition(ps, [(
        _tc("run_command", {"command": "pytest scico/test/linop/xray/test_xray_2d.py"}),
        {"success": True, "data": {
            "exit_code": 0,
            "test_summary": {"passed": 0, "failed": 1},
        }},
    )])
    assert ps.f2p_files_covered == set()
    assert ps.test_passed_after_edit is False


# ── check_finish_plan_gate: F5 missing-file message ─────────────────


def test_gate_message_lists_missing_f2p_files():
    """When some F2P files are covered and some aren't, the gate should
    name the missing ones — generic 'run failing tests' wording was
    why v27.1 let scico-561 ship after fixing only 2D."""
    ps = _ps_with_edit()
    ps.phase = "verify"  # gate only fires in verify
    ps.f2p_files_covered.add("scico/test/linop/xray/test_xray_2d.py")
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(ps, dispatched, F2P, intercepts_so_far=0)
    assert msg, "gate should fire when test_passed_after_edit is False"
    assert "test_xray_3d.py" in msg, (
        "missing-file gate must name the unrun file explicitly"
    )
    assert "1/2" in msg or "1 of 2" in msg.lower(), (
        "gate message should show coverage progress"
    )


def test_gate_releases_after_one_intercept():
    """Single-shot — second finish_plan attempt is allowed through so
    truly broken environments can still terminate."""
    ps = _ps_with_edit()
    ps.phase = "verify"
    ps.f2p_files_covered.add("scico/test/linop/xray/test_xray_2d.py")
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(ps, dispatched, F2P, intercepts_so_far=1)
    assert msg == "", "gate must release after one intercept"


def test_gate_silent_when_full_coverage():
    """If every F2P file has been exercised (and tests pass), no nudge."""
    ps = _ps_with_edit()
    ps.phase = "verify"
    ps.f2p_files_covered = set(distinct_f2p_files(F2P))
    ps.test_passed_after_edit = True
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(ps, dispatched, F2P, intercepts_so_far=0)
    assert msg == ""


# ── v5 partial-pass gate ────────────────────────────────────────────


def _verify_ps() -> PhaseState:
    """Phase state in verify phase with one prior edit (no F2P pass)."""
    ps = _ps_with_edit()
    ps.phase = "verify"
    return ps


def test_gate_intercepts_twice_with_max_intercepts_2():
    """v5: bumped to max_intercepts=2 — gate fires on attempts 1 and 2,
    releases on attempt 3."""
    ps = _verify_ps()
    failures = [{"test": "tests/test_calendar.py::test_holiday", "error": "AssertionError"}]
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    # First two intercepts should fire.
    msg0 = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=0,
        max_intercepts=2, last_f2p_failures=failures,
    )
    assert msg0, "1st attempt must intercept"
    msg1 = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=1,
        max_intercepts=2, last_f2p_failures=failures,
    )
    assert msg1, "2nd attempt must still intercept under max_intercepts=2"
    # Third attempt: gate releases.
    msg2 = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=2,
        max_intercepts=2, last_f2p_failures=failures,
    )
    assert msg2 == "", "gate must release after max_intercepts hit"


def test_gate_message_lists_failing_f2p_tests():
    """v5: partial-pass branch should enumerate failing test IDs and
    their assertion errors so the model can target a focused fix."""
    ps = _verify_ps()
    failures = [
        {"test": "tests/test_a.py::test_one", "error": "AssertionError: 1 != 2"},
        {"test": "tests/test_b.py::test_two", "error": "ValueError: bad input"},
    ]
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=0,
        max_intercepts=2, last_f2p_failures=failures,
    )
    assert msg, "partial-pass gate should fire"
    assert "tests/test_a.py::test_one" in msg
    assert "tests/test_b.py::test_two" in msg
    assert "AssertionError: 1 != 2" in msg
    assert "ValueError: bad input" in msg


def test_gate_falls_back_when_no_failure_detail():
    """v5: when last_f2p_failures is empty, fall back to existing
    F5 missing-files / generic message paths so non-pytest runners
    still get useful guidance."""
    ps = _verify_ps()
    # Simulate partial F5 coverage so the missing-file branch fires.
    ps.f2p_files_covered.add("scico/test/linop/xray/test_xray_2d.py")
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=0,
        max_intercepts=2, last_f2p_failures=[],
    )
    assert msg, "fallback path should still fire"
    # Should be the F5 missing-files message, not the v5 partial-pass one.
    assert "test_xray_3d.py" in msg
    # Confirm the v5 partial-pass framing is absent.
    assert "still failing" not in msg.lower() or "exercised yet" in msg


def test_gate_releases_after_max_intercepts_partial_pass():
    """v5: safety release — even with last_f2p_failures populated, the
    gate releases on the 3rd attempt with max_intercepts=2 so a
    structurally unrunnable environment cannot trap the agent."""
    ps = _verify_ps()
    failures = [{"test": "tests/test_x.py::test_y", "error": "ImportError"}]
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=2,
        max_intercepts=2, last_f2p_failures=failures,
    )
    assert msg == "", "gate must release at the intercept budget"


# ── v6b collection-error gate ───────────────────────────────────────


def test_gate_collection_error_branch_fires():
    """v6b: when last_f2p_collection_error=True and last_f2p_failures
    is empty, gate fires with a "crashed at collection" message that
    tells the agent to fix imports / deps before finishing."""
    ps = _verify_ps()
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=0,
        max_intercepts=2, last_f2p_failures=[],
        last_f2p_collection_error=True,
    )
    assert msg, "v6b collection-error branch should fire"
    assert "crashed at collection" in msg
    assert "import" in msg.lower()


def test_gate_partial_pass_takes_precedence_over_collection_error():
    """v6b: when both per-test failures AND a collection-error flag are
    present, the partial-pass branch should win (more actionable)."""
    ps = _verify_ps()
    failures = [{"test": "tests/test_a.py::test_one", "error": "AssertionError: x"}]
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=0,
        max_intercepts=2, last_f2p_failures=failures,
        last_f2p_collection_error=True,
    )
    assert msg, "gate should fire"
    assert "tests/test_a.py::test_one" in msg, "partial-pass branch must win"
    assert "crashed at collection" not in msg, (
        "collection-error branch should NOT fire when per-test failures present"
    )


def test_gate_collection_error_releases_after_max_intercepts():
    """v6b: safety release — collection-error branch also bounded by
    max_intercepts so a permanently-broken test env cannot trap the
    agent."""
    ps = _verify_ps()
    dispatched = [(
        _tc("finish_plan", {"status": "success"}),
        {"success": True, "data": {}},
    )]
    msg = check_finish_plan_gate(
        ps, dispatched, F2P, intercepts_so_far=2,
        max_intercepts=2, last_f2p_failures=[],
        last_f2p_collection_error=True,
    )
    assert msg == "", "gate must release at the intercept budget"
