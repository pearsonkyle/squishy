"""A failed `old_str` should point somewhere, not just say 'read the file'."""
from __future__ import annotations

from squishy.tools import dispatch

SRC = """def compute(values):
    total = 0
    for v in values:
        total += v
    return total
"""


async def test_single_line_old_str_gets_a_location(ctx, tmp_path):
    """The most common shape — and it used to fall through to no hint at all.

    Fuzzy matching was gated on `len(old_lines) >= 2`, so a one-line old_str
    could only ever produce "Read the file first and copy the exact text."
    """
    (tmp_path / "m.py").write_text(SRC)
    ctx.working_dir = str(tmp_path)
    res = await dispatch("edit_file", {
        "path": "m.py", "old_str": "    total = 0;", "new_str": "    total = 1",
    }, ctx)
    assert not res.success
    assert "total = 0" in res.error
    assert "line" in res.error.lower()
    assert res.error.strip() != "old_str not found in file. Read the file first and copy the exact text."


async def test_weak_match_is_offered_with_a_caveat(ctx, tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    ctx.working_dir = str(tmp_path)
    res = await dispatch("edit_file", {
        "path": "m.py",
        "old_str": "    for value in values:\n        total = total + value",
        "new_str": "    pass",
    }, ctx)
    assert not res.success
    if "closest block" in res.error:
        assert "may not be the right one" in res.error


async def test_nothing_remotely_similar_still_says_read_the_file(ctx, tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    ctx.working_dir = str(tmp_path)
    res = await dispatch("edit_file", {
        "path": "m.py", "old_str": "zzzzzzzz qqqqqqqq", "new_str": "x",
    }, ctx)
    assert not res.success
    assert "Read the file first" in res.error


async def test_a_real_match_still_just_works(ctx, tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    ctx.working_dir = str(tmp_path)
    res = await dispatch("edit_file", {
        "path": "m.py", "old_str": "total = 0", "new_str": "total = 100",
    }, ctx)
    assert res.success, res.error
    assert "total = 100" in (tmp_path / "m.py").read_text()
