"""Tests for agent_state helpers: command classification, problem extraction, test coverage."""
from __future__ import annotations

from squishy.agent_state import (
    extract_problem_files,
    is_test_command,
    path_matches_problem,
    test_covers_fail_to_pass as _test_covers_fail_to_pass,
)


# -- _test_covers_fail_to_pass ------------------------------------------------

class TestCoversFailToPass:
    def test_empty_list_returns_true(self):
        assert _test_covers_fail_to_pass("pytest tests/", []) is True

    def test_full_test_id_match(self):
        fail = ["tests/test_foo.py::TestBar::test_baz"]
        assert _test_covers_fail_to_pass(
            "pytest tests/test_foo.py::TestBar::test_baz -xvs", fail
        ) is True

    def test_file_only_match(self):
        fail = ["tests/test_foo.py::TestBar::test_baz"]
        assert _test_covers_fail_to_pass("pytest tests/test_foo.py -x", fail) is True

    def test_module_match(self):
        fail = ["tests/test_foo.py::test_baz"]
        assert _test_covers_fail_to_pass("pytest tests/test_foo -x", fail) is True

    def test_no_match_returns_false(self):
        fail = ["tests/test_foo.py::test_baz"]
        assert _test_covers_fail_to_pass("pytest tests/test_bar.py", fail) is False

    def test_backslash_normalization(self):
        fail = ["tests\\test_foo.py::test_baz"]
        assert _test_covers_fail_to_pass("pytest tests/test_foo.py", fail) is True

    def test_multiple_fail_to_pass_any_match(self):
        fail = ["tests/test_a.py::test_1", "tests/test_b.py::test_2"]
        assert _test_covers_fail_to_pass("pytest tests/test_b.py", fail) is True

    def test_multiple_fail_to_pass_none_match(self):
        fail = ["tests/test_a.py::test_1", "tests/test_b.py::test_2"]
        assert _test_covers_fail_to_pass("pytest tests/test_c.py", fail) is False


# -- is_test_command -----------------------------------------------------------

class TestIsTestCommand:
    def test_pytest(self):
        assert is_test_command("pytest tests/ -x") is True

    def test_python_m_pytest(self):
        assert is_test_command("python -m pytest tests/test_foo.py") is True

    def test_unittest(self):
        assert is_test_command("python -m unittest tests.test_foo") is True

    def test_test_underscore(self):
        assert is_test_command("python test_something.py") is True

    def test_non_test(self):
        assert is_test_command("python setup.py install") is False

    def test_non_test_command(self):
        assert is_test_command("pip install requests") is False


# -- extract_problem_files -----------------------------------------------------

class TestExtractProblemFiles:
    def test_py_path(self):
        text = "Error in django/db/models/query.py line 42"
        result = extract_problem_files(text)
        assert "django/db/models/query.py" in result

    def test_module_dotted(self):
        text = "ModuleNotFoundError: django.db.models.query"
        result = extract_problem_files(text)
        # Should convert dotted module to file path
        assert any("django/db/models/query.py" in p for p in result)

    def test_multiple_paths(self):
        text = "File utils/helper.py and core/engine.py"
        result = extract_problem_files(text)
        assert "utils/helper.py" in result
        assert "core/engine.py" in result

    def test_no_paths(self):
        text = "Something went wrong"
        result = extract_problem_files(text)
        assert len(result) == 0


# -- path_matches_problem -----------------------------------------------------

class TestPathMatchesProblem:
    def test_exact_match(self):
        assert path_matches_problem("src/utils.py", {"src/utils.py"}) is True

    def test_partial_match(self):
        assert path_matches_problem("full/path/src/utils.py", {"src/utils.py"}) is True

    def test_basename_match(self):
        assert path_matches_problem("somewhere/utils.py", {"other/path/utils.py"}) is True

    def test_no_match(self):
        assert path_matches_problem("different/module.py", {"src/utils.py"}) is False

    def test_backslash_normalized(self):
        assert path_matches_problem("src\\utils.py", {"src/utils.py"}) is True


# -- #4: pytest-substring false positives --------------------------------------

def test_is_test_command_ignores_pytest_in_commit_message():
    from squishy.agent_state import is_test_command
    assert is_test_command('git commit -m "fix pytest failure"') is False
    assert is_test_command("echo running pytest soon") is False
    assert is_test_command("grep pytest setup.cfg") is False


def test_f2p_coverage_rejects_non_run_pytest():
    from squishy.agent_state import f2p_files_in_command
    f2p = ["tests/a.py::t1", "tests/b.py::t2"]
    assert f2p_files_in_command('git commit -m "pytest"', f2p) == set()
    assert f2p_files_in_command("pytest --version", f2p) == set()
    assert f2p_files_in_command("pytest --collect-only", f2p) == set()
    # A real bare run still covers the whole suite.
    assert f2p_files_in_command("pytest -x", f2p) == {"tests/a.py", "tests/b.py"}
