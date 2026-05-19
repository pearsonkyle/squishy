"""Tests for scripts/rebench_eval/run_eval.py:parse_pytest_output.

Covers v27 audit items A1 + A4 — eval_status must distinguish env failures
from test failures so SFT training-data filtering and honest scoring work.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the script as a module (it lives outside the package).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "rebench_eval" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("rebench_run_eval", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_parse_pytest_output_clean_pass():
    output = "PASSED tests/test_foo.py::test_bar\n1 passed in 0.01s"
    res = _mod.parse_pytest_output(
        output, ["tests/test_foo.py::test_bar"], [],
    )
    assert res["fail_to_pass"]["passed"] == 1
    assert res["fail_to_pass"]["total"] == 1
    assert res["eval_status"] == "ok"


def test_parse_pytest_output_test_patch_failure_is_eval_error():
    output = "EVAL_ERROR_TEST_PATCH\nbash: exit 2"
    res = _mod.parse_pytest_output(
        output, ["tests/test_foo.py::test_bar"], [],
    )
    assert res["eval_status"] == "eval_error"
    assert "test_patch" in res["eval_error"]


def test_parse_pytest_output_reinstall_failure_is_eval_error():
    output = "EVAL_ERROR_REINSTALL\npip install failed"
    res = _mod.parse_pytest_output(
        output, ["tests/test_foo.py::test_bar"], [],
    )
    assert res["eval_status"] == "eval_error"
    assert "re-install" in res["eval_error"].lower()


def test_parse_pytest_output_pre_install_failure_is_eval_error():
    output = "EVAL_ERROR_PRE_INSTALL\nexit 3"
    res = _mod.parse_pytest_output(
        output, ["tests/test_foo.py::test_bar"], [],
    )
    assert res["eval_status"] == "eval_error"


def test_parse_pytest_output_module_not_found_is_eval_error():
    """All-NOT_FOUND F2P plus ModuleNotFoundError → eval_error, not test_failure.

    This is the exact scenario that broke the v26 agentdojo evaluation.
    """
    output = "ModuleNotFoundError: No module named 'agentdojo'\n"
    res = _mod.parse_pytest_output(
        output, ["tests/test_x.py::test_y"], [],
    )
    assert res["fail_to_pass"]["passed"] == 0
    assert res["eval_status"] == "eval_error"
    assert "import" in res["eval_error"].lower() or "env" in res["eval_error"].lower()


def test_parse_pytest_output_real_test_failure_stays_test_failure():
    """A genuine assertion failure should NOT be flagged as eval_error."""
    output = (
        "FAILED tests/test_foo.py::test_bar - AssertionError\n"
        "1 failed in 0.02s"
    )
    res = _mod.parse_pytest_output(
        output, ["tests/test_foo.py::test_bar"], [],
    )
    assert res["fail_to_pass"]["passed"] == 0
    assert res["eval_status"] == "ok"


def test_parse_pytest_output_reinstall_overridden_when_all_pass():
    """v6d: REINSTALL flag is cosmetic when tests already ran cleanly.

    Reproduces brutils-116 from the v6c run: tests passed end-to-end but
    the post-test reinstall hit /testbed bind quirks, marking eval_error
    and blocking resolved=True for a patch that actually worked.
    """
    output = (
        "PASSED tests/test_foo.py::test_bar\n"
        "PASSED tests/test_foo.py::test_p1\n"
        "PASSED tests/test_foo.py::test_p2\n"
        "3 passed in 0.05s\n"
        "EVAL_ERROR_REINSTALL\n"
        "EnvironmentNameNotFound: testbed\n"
    )
    res = _mod.parse_pytest_output(
        output,
        ["tests/test_foo.py::test_bar"],
        ["tests/test_foo.py::test_p1", "tests/test_foo.py::test_p2"],
    )
    assert res["fail_to_pass"]["passed"] == 1
    assert res["pass_to_pass"]["passed"] == 2
    assert res["pass_to_pass"]["checked"] == 2
    assert res["eval_status"] == "ok", \
        "REINSTALL must be overridden when all F2P + P2P passed"
    assert res["eval_error"] == ""


def test_parse_pytest_output_reinstall_not_overridden_when_p2p_fails():
    """v6d: REINSTALL override must NOT mask a real P2P regression.

    If the patch fixes F2P but breaks an unrelated P2P test, the
    override must not fire — that's a genuine quality issue we want to
    surface, not hide behind a reinstall hiccup.
    """
    output = (
        "PASSED tests/test_foo.py::test_bar\n"
        "PASSED tests/test_foo.py::test_p1\n"
        "FAILED tests/test_foo.py::test_p2 - AssertionError\n"
        "2 passed, 1 failed in 0.05s\n"
        "EVAL_ERROR_REINSTALL\n"
    )
    res = _mod.parse_pytest_output(
        output,
        ["tests/test_foo.py::test_bar"],
        ["tests/test_foo.py::test_p1", "tests/test_foo.py::test_p2"],
    )
    assert res["fail_to_pass"]["passed"] == 1
    assert res["pass_to_pass"]["passed"] == 1  # one P2P failed
    assert res["eval_status"] == "eval_error"
    assert "re-install" in res["eval_error"].lower()


# -- v6e/5: stderr-capture invariance -----------------------------------------

def test_parse_pytest_output_test_patch_with_stderr_block():
    """v6e: the new git-apply stderr capture wraps EVAL_ERROR_TEST_PATCH
    with a stderr block; the parser substring match must still detect
    the sentinel.  Regression guard for the v6e/5 eval-script change."""
    output = (
        "EVAL_ERROR_TEST_PATCH\n"
        "--- test_patch apply stderr ---\n"
        "error: patch failed: tests/test_foo.py:42\n"
        "error: tests/test_foo.py: patch does not apply\n"
        "--- end test_patch apply stderr ---\n"
    )
    res = _mod.parse_pytest_output(
        output,
        ["tests/test_foo.py::test_bar"],
        [],
    )
    assert res["eval_status"] == "eval_error"
    assert "test_patch" in res["eval_error"].lower()
    # F2P collection should be empty (tests never ran).
    assert res["fail_to_pass"]["passed"] == 0


def test_parse_pytest_output_model_patch_stderr_does_not_break_parsing():
    """v6e: 'ERROR: model_patch apply failed' alone is not a sentinel
    we map to eval_status — it just exits 1 and the test command never
    runs.  Confirm the surrounding stderr block doesn't accidentally
    flip eval_status (no false positive on EVAL_ERROR_* substrings)."""
    output = (
        "ERROR: model_patch apply failed\n"
        "--- model_patch apply stderr ---\n"
        "error: while searching for:\n  some context line\n"
        "error: patch failed: src/foo.py:10\n"
        "--- end model_patch apply stderr ---\n"
    )
    res = _mod.parse_pytest_output(
        output,
        ["tests/test_foo.py::test_bar"],
        [],
    )
    # No EVAL_ERROR_* sentinel present; status defaults to "ok" with
    # zero passed counts (the harness wraps this case separately).
    assert res["eval_status"] == "ok"
    assert res["fail_to_pass"]["passed"] == 0
