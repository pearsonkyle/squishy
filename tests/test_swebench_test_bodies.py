"""Tests for `_extract_test_context_bodies` (F2).

The SWE-bench harness runs the agent against the *pre-patch* tree, so
the FAIL_TO_PASS test functions don't yet exist in the workspace —
they are added by `test_patch` at evaluation time.  v27's B1 helper
returned `[]` in this case, leaving the prompt with zero test-body
context.  v27.1 falls back to sibling tests in the same file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from squishy.bench.swebench import (
    _common_prefix_len,
    _extract_test_context_bodies,
)


def test_common_prefix_len_basic():
    assert _common_prefix_len("test_foo_bar", "test_foo_baz") == len("test_foo_ba")
    assert _common_prefix_len("test_foo", "test_bar") == len("test_")
    assert _common_prefix_len("abc", "xyz") == 0
    assert _common_prefix_len("same", "same") == 4


def test_extract_returns_real_body_when_function_exists(tmp_path: Path):
    """When the F2P function IS in the workspace (Terminal-bench / unit
    tests), use that body directly — backwards compat."""
    (tmp_path / "tests").mkdir()
    test_file = tmp_path / "tests" / "test_widget.py"
    test_file.write_text(
        "def test_widget_grows():\n"
        "    w = Widget()\n"
        "    w.grow()\n"
        "    assert w.size == 2\n"
    )
    out = _extract_test_context_bodies(
        tmp_path, ["tests/test_widget.py::test_widget_grows"],
    )
    assert len(out) == 1
    label, body, is_sibling = out[0]
    assert label == "tests/test_widget.py::test_widget_grows"
    assert is_sibling is False
    assert "def test_widget_grows" in body
    assert "assert w.size == 2" in body


def test_extract_falls_back_to_siblings_when_function_missing(tmp_path: Path):
    """SWE-bench case: F2P function not in the workspace.  We should
    return up to 2 siblings, picked by longest shared name prefix."""
    (tmp_path / "tests").mkdir()
    test_file = tmp_path / "tests" / "test_widget.py"
    test_file.write_text(
        "def test_widget_grows():\n"
        "    w = Widget()\n"
        "    w.grow()\n"
        "    assert w.size == 2\n"
        "\n"
        "def test_widget_shrinks():\n"
        "    w = Widget()\n"
        "    w.shrink()\n"
        "    assert w.size == 0\n"
        "\n"
        "def test_unrelated_helper():\n"
        "    assert True\n"
    )
    # F2P refers to a function that doesn't exist yet.
    out = _extract_test_context_bodies(
        tmp_path, ["tests/test_widget.py::test_widget_clones"],
    )
    assert len(out) >= 1
    labels = [label for label, _body, _is_sib in out]
    bodies = [body for _label, body, _is_sib in out]
    assert all(is_sib for _label, _body, is_sib in out), (
        "all returned bodies should be flagged as siblings"
    )
    # The longest-prefix sibling is `test_widget_grows` or
    # `test_widget_shrinks` (both share `test_widget_`); both are valid.
    assert any("sibling of test_widget_clones" in lbl for lbl in labels)
    body_text = "\n".join(bodies)
    assert "Widget()" in body_text
    # The unrelated helper has a much shorter shared prefix and SHOULD
    # NOT make it in unless we pick more than 2 siblings.
    if len(out) <= 2:
        assert "test_unrelated_helper" not in body_text


def test_extract_returns_empty_when_file_missing(tmp_path: Path):
    out = _extract_test_context_bodies(
        tmp_path, ["tests/does_not_exist.py::test_foo"],
    )
    assert out == []


def test_extract_returns_empty_when_no_test_functions(tmp_path: Path):
    """File exists but has no test_* functions → no useful siblings."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        "import pytest\n"
        "\n"
        "@pytest.fixture\n"
        "def helper():\n"
        "    return 42\n"
    )
    out = _extract_test_context_bodies(
        tmp_path, ["tests/conftest.py::test_thing"],
    )
    assert out == []


def test_extract_caps_total_lines(tmp_path: Path):
    """Global cap of 60 lines must be honored even when many siblings
    are available."""
    (tmp_path / "tests").mkdir()
    sibs = "\n\n".join(
        f"def test_widget_v{i}():\n"
        + "\n".join(f"    line_{j} = {j}" for j in range(20))
        for i in range(10)
    )
    (tmp_path / "tests" / "test_widget.py").write_text(sibs)
    out = _extract_test_context_bodies(
        tmp_path, ["tests/test_widget.py::test_widget_missing"],
        max_total_lines=30,
    )
    total_lines = sum(b.count("\n") + 1 for _l, b, _s in out)
    assert total_lines <= 30 + 5  # +5 for the truncation marker fudge
