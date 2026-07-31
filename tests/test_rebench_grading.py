"""Grading and prompt assembly for the in-container SWE-rebench runner.

The grader used to score `resolved = post.exit_code == 0` — the whole test
command had to come back green. SWE-bench asks something narrower: did every
FAIL_TO_PASS test pass, and did any PASS_TO_PASS test that passed before now
fail. Running the *gold* patch through the old grader on a 5-instance
validation set scored 2 of 4 as failures, purely because unrelated tests in
the same file were already red. Every resolve rate measured under it was
understated by an unknown amount.

These tests also just import the script. A module-level NameError there costs
a full container run to discover — which is exactly what happened when an edit
swallowed EMPTY_PATCH_NUDGE.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "rebench_container" / "run_bench.py"
)


@pytest.fixture(scope="module")
def rb():
    if not _SCRIPT.exists():
        pytest.skip("rebench container runner not present")
    spec = importlib.util.spec_from_file_location("run_bench", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_module_level_names_all_resolve(rb):
    """Catches the NameError class of bug without spending a container."""
    assert rb.EMPTY_PATCH_NUDGE
    assert rb.PROMPT.format(problem="p", tests=rb._tests_block(["a.py::b"]))


# -- status parsing ----------------------------------------------------------

def test_parses_pytest_short_summary(rb):
    out = """
PASSED tests/test_a.py::test_one
FAILED tests/test_a.py::test_two
ERROR tests/test_b.py::test_three
"""
    assert rb._parse_statuses(out) == {
        "tests/test_a.py::test_one": "PASSED",
        "tests/test_a.py::test_two": "FAILED",
        "tests/test_b.py::test_three": "ERROR",
    }


def test_parses_go_verbose_output(rb):
    out = """
=== RUN   TestFoo
--- PASS: TestFoo (0.00s)
=== RUN   TestBar
--- FAIL: TestBar (0.01s)
"""
    assert rb._parse_statuses(out) == {"TestFoo": "PASSED", "TestBar": "FAILED"}


def test_parametrized_ids_match_in_either_direction(rb):
    assert rb._lookup({"a.py::t[case-3]": "PASSED"}, "a.py::t") == "PASSED"
    assert rb._lookup({"a.py::t": "PASSED"}, "a.py::t[case-3]") == "PASSED"


# -- the criterion -----------------------------------------------------------

def test_unrelated_pre_existing_failure_does_not_block_resolve(rb):
    """The bug this whole change exists for.

    `already_red` failed before the patch and still fails after. It is not the
    model's to fix, and it must not veto a correct patch — but under exit-code
    grading it did, because the suite never returns 0.
    """
    pre = {"statuses": {"a::target": "FAILED", "a::keep": "PASSED",
                        "a::already_red": "FAILED"}}
    post = {"statuses": {"a::target": "PASSED", "a::keep": "PASSED",
                         "a::already_red": "FAILED"}}
    got = rb._grade_by_test(pre, post, ["a::target"], ["a::keep", "a::already_red"])
    assert got["resolved"] is True
    assert got["grade_method"] == "per-test"


def test_genuine_regression_blocks_resolve(rb):
    pre = {"statuses": {"a::target": "FAILED", "a::keep": "PASSED"}}
    post = {"statuses": {"a::target": "PASSED", "a::keep": "FAILED"}}
    got = rb._grade_by_test(pre, post, ["a::target"], ["a::keep"])
    assert got["resolved"] is False
    assert got["p2p_regressed"] == ["a::keep"]


def test_target_still_failing_blocks_resolve(rb):
    pre = {"statuses": {"a::target": "FAILED"}}
    post = {"statuses": {"a::target": "FAILED"}}
    got = rb._grade_by_test(pre, post, ["a::target"], [])
    assert got["resolved"] is False
    assert got["f2p_failing"] == ["a::target"]


def test_unparseable_output_falls_back_to_exit_code(rb):
    """Maven/Gradle print no per-test lines; the caller must know to use exit code."""
    assert rb._grade_by_test({"statuses": {}}, {"statuses": {}}, ["x"], []) is None


# -- base gate ---------------------------------------------------------------

def test_base_gate_is_asked_per_test(rb):
    """Wrong in both directions before: a red neighbour made a gradable
    instance look ungradable, and a green target hid behind a red neighbour."""
    # Targets red, an unrelated test red too -> still a usable instance.
    assert not rb._already_green_at_base(
        {"statuses": {"a::t": "FAILED", "a::x": "FAILED"}, "exit_code": 1}, ["a::t"])
    # Targets already green -> nothing to prove, not usable.
    assert rb._already_green_at_base(
        {"statuses": {"a::t": "PASSED", "a::x": "FAILED"}, "exit_code": 1}, ["a::t"])


# -- prompt assembly ---------------------------------------------------------

def test_tests_block_leads_with_the_per_file_spread(rb):
    """wtforms-614 has 262 ids over 3 files; a prefix of 10 hid where the work was."""
    ids = (
        [f"tests/test_validators.py::t{i}" for i in range(146)]
        + [f"tests/test_fields.py::t{i}" for i in range(97)]
        + [f"tests/test_widgets.py::t{i}" for i in range(19)]
    )
    block = rb._tests_block(ids)
    assert "262 tests" in block
    assert "146 in tests/test_validators.py" in block
    # Every file that carries part of the task appears in the sample.
    for f in ("test_validators.py", "test_fields.py", "test_widgets.py"):
        assert f in block


def test_tests_block_is_empty_without_targets(rb):
    assert rb._tests_block([]) == ""
