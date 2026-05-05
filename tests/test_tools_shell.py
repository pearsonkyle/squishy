from __future__ import annotations
 
 
from squishy.tools.shell import _extract_test_failures, run_command
 
 
 
async def test_run_command_success(ctx):
    r = await run_command.run({"command": "echo hello"}, ctx)
    assert r.success
    assert r.data["exit_code"] == 0
    assert "hello" in r.data["stdout"]
 
 
async def test_run_command_nonzero_exit(ctx):
    r = await run_command.run({"command": "sh -c 'exit 7'"}, ctx)
    # Non-zero exit now reports as tool failure so a 7B model can't mistake a
    # crashing command for success, but exit_code and outputs stay in `data`.
    assert not r.success
    assert r.data["exit_code"] == 7
    assert "exited 7" in r.error


async def test_run_command_truncation_is_signalled(ctx):
    # Produce ~20KB of stdout to trip the OUTPUT_CAP_STDOUT limit.
    r = await run_command.run(
        {"command": "python -c \"print('x' * 20000)\""}, ctx
    )
    assert r.success
    assert r.data["truncated"] is True
    assert "truncated" in r.data["stdout"]
 
 
async def test_run_command_timeout(ctx):
    r = await run_command.run({"command": "sleep 5", "timeout": 1}, ctx)
    assert not r.success
    assert "timed out" in r.error
 
 
async def test_run_command_cwd(ctx, tmp_path):
    (tmp_path / "marker").write_text("here")
    r = await run_command.run({"command": "ls", "cwd": str(tmp_path)}, ctx)
    assert r.success
    assert "marker" in r.data["stdout"]


# -- _extract_test_failures tests -------------------------------------------

PYTEST_OUTPUT_WITH_FAILURES = """\
============================= test session starts ==============================
collected 10 items

tests/test_foo.py::test_bar PASSED
tests/test_foo.py::test_baz FAILED
tests/test_foo.py::test_qux FAILED

=================================== FAILURES ===================================
_____________________________ test_baz ______________________________

    def test_baz():
>       assert 1 == 2
E       AssertionError: assert 1 == 2

_____________________________ test_qux ______________________________

    def test_qux():
>       result = foo()
E       TypeError: 'NoneType' object is not callable

=========================== short test summary info ============================
FAILED tests/test_foo.py::test_baz - AssertionError: assert 1 == 2
FAILED tests/test_foo.py::test_qux - TypeError: 'NoneType' object is not callable
============================== 2 failed, 8 passed in 1.23s ====================
"""

PYTEST_ALL_PASSING = """\
============================= test session starts ==============================
collected 5 items

tests/test_foo.py .....                                                  [100%]

============================== 5 passed in 0.45s ===============================
"""

PYTEST_COLLECTION_ERROR = """\
============================= test session starts ==============================
ERRORS during collection
ERROR collecting tests/test_foo.py
ImportError while importing test module
E   ModuleNotFoundError: No module named 'foo'
short test summary info
ERROR tests/test_foo.py
============================== 1 error in 0.12s ================================
"""

PYTEST_SUMMARY_ONLY = """\
FAILED tests/test_a.py::test_x - ValueError: bad value
FAILED tests/test_a.py::test_y - KeyError: 'missing'
===== 2 failed, 10 passed in 3.5s =====
"""


def test_extract_failures_with_summary():
    result = _extract_test_failures(PYTEST_OUTPUT_WITH_FAILURES)
    assert result is not None
    assert result["passed"] == 8
    assert result["failed"] == 2
    assert len(result["failures"]) == 2
    assert result["failures"][0]["test"] == "tests/test_foo.py::test_baz"
    assert "AssertionError" in result["failures"][0]["error"]
    assert result["failures"][1]["test"] == "tests/test_foo.py::test_qux"
    assert "TypeError" in result["failures"][1]["error"]


def test_extract_failures_all_passing():
    """All-passing output with only 'passed' marker is below _looks_like_pytest
    threshold (2 markers needed). Returns None since there's nothing useful to
    extract — no failures to report."""
    result = _extract_test_failures(PYTEST_ALL_PASSING)
    # Only has 1 marker ("passed"), so _looks_like_pytest returns False.
    assert result is None


def test_extract_failures_collection_error():
    """Collection-only errors don't have enough markers for _looks_like_pytest.
    These are handled by goal drift detection instead."""
    result = _extract_test_failures(PYTEST_COLLECTION_ERROR)
    assert result is None


def test_extract_failures_summary_only():
    result = _extract_test_failures(PYTEST_SUMMARY_ONLY)
    assert result is not None
    assert result["failed"] == 2
    assert result["passed"] == 10
    assert len(result["failures"]) == 2
    assert result["failures"][0]["test"] == "tests/test_a.py::test_x"


def test_extract_failures_non_pytest():
    result = _extract_test_failures("hello world\njust some random output")
    assert result is None


def test_extract_failures_fallback_to_headers():
    """When no short test summary, parse FAILURES section headers."""
    output = """\
= FAILURES =
_____ test_something _____

    assert False

= 1 failed in 0.1s =
"""
    result = _extract_test_failures(output)
    assert result is not None
    assert result["failed"] == 1
    assert len(result["failures"]) == 1
    assert result["failures"][0]["test"] == "test_something"
