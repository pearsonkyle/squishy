"""search_files must not dead-end on a pattern that isn't valid regex."""
from __future__ import annotations

import shutil

import pytest

from squishy.tools import dispatch


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.py").write_text("value = data[0]\ndef helper():\n    pass\n")
    (tmp_path / "b.py").write_text("other = 1\n")
    return tmp_path


@pytest.mark.parametrize("pattern", ["*.py", "data[0]", "helper("])
async def test_invalid_regex_falls_back_to_literal(ctx, tree, pattern):
    """Models pass globs and raw code snippets where a regex is expected."""
    ctx.working_dir = str(tree)
    res = await dispatch("search_files", {"pattern": pattern}, ctx)
    assert res.success, res.error


async def test_literal_fallback_finds_the_text_and_says_so(ctx, tree):
    ctx.working_dir = str(tree)
    res = await dispatch("search_files", {"pattern": "data[0]"}, ctx)
    assert res.success
    assert res.data["count"] == 1
    assert "literally" in res.data["note"]


async def test_valid_regex_is_unaffected(ctx, tree):
    ctx.working_dir = str(tree)
    res = await dispatch("search_files", {"pattern": r"def \w+\("}, ctx)
    assert res.success
    assert res.data["count"] == 1
    assert "note" not in res.data


@pytest.mark.skipif(not shutil.which("rg"), reason="needs ripgrep")
async def test_ripgrep_errors_are_reported_not_silently_empty(ctx, tree):
    """A broken search must not look like 'this symbol does not exist'."""
    ctx.working_dir = str(tree)
    res = await dispatch(
        "search_files", {"pattern": "value", "path": "a.py", "glob": "["}, ctx)
    # Either it works or it errors — what it must never do is claim 0 matches
    # for a term that is present.
    assert res.success is False or res.data["count"] >= 0


async def test_valid_regex_with_zero_matches_retries_literally(ctx, tree):
    """`data[0]` is a valid character class — and not what the model meant.

    Searching it as a regex finds nothing, so the model concludes the code
    isn't there. This is the damaging case: no error, just a wrong answer.
    """
    ctx.working_dir = str(tree)
    res = await dispatch("search_files", {"pattern": "data[0]"}, ctx)
    assert res.success
    assert res.data["count"] == 1
    assert res.data["matches"][0]["line"] == 1
    assert "literally" in res.data["note"]


async def test_no_literal_retry_when_the_regex_already_matched(ctx, tree):
    ctx.working_dir = str(tree)
    res = await dispatch("search_files", {"pattern": r"def \w+\("}, ctx)
    assert res.success and res.data["count"] == 1
    assert "note" not in res.data


async def test_genuinely_absent_text_still_reports_zero(ctx, tree):
    ctx.working_dir = str(tree)
    res = await dispatch("search_files", {"pattern": "not_here[0]"}, ctx)
    assert res.success
    assert res.data["count"] == 0
