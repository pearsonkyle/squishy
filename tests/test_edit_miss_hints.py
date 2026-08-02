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


async def test_a_failed_edit_names_the_symbol_s_real_lines(tmp_path):
    """The dead-end case, answered from the graph.

    qiskit-terra-5662's only edit attempt was at turn 73, searching for
    `def _get_measure_link(self, qubit, clbit):` — a signature that had
    changed. The fuzzy matcher found nothing, the hint was "read the file
    first" (which it had done five times), and it never tried to edit again.
    A run of 100 turns produced no patch off that one dead end.
    """
    from squishy.graph import build_repo_graph
    from squishy.tools import dispatch
    from squishy.tools.base import ToolContext

    (tmp_path / "core.py").write_text(
        "class Canvas:\n"
        "    def _get_measure_link(self, qubit):\n"
        "        return qubit\n"
        "\n"
        "    def other(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    build_repo_graph(tmp_path)
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="yolo",
                      use_sandbox=False)

    res = await dispatch("edit_file", {
        "path": "core.py",
        # The real signature has no `clbit`, so no fuzzy block will match.
        "old_str": "def _get_measure_link(self, qubit, clbit):\n"
                   "        raise NotImplementedError\n"
                   "        # padding to defeat the fuzzy matcher entirely\n",
        "new_str": "x",
    }, ctx)

    assert not res.success
    assert "_get_measure_link" in res.error
    assert "lines 2-3" in res.error, res.error
    assert "read_file" in res.error


async def test_the_hint_degrades_without_a_graph(tmp_path):
    """No graph must mean the old advice, not a broken message."""
    from squishy.tools import dispatch
    from squishy.tools.base import ToolContext

    (tmp_path / "core.py").write_text("a = 1\n", encoding="utf-8")
    ctx = ToolContext(working_dir=str(tmp_path), permission_mode="yolo",
                      use_sandbox=False)
    res = await dispatch("edit_file", {
        "path": "core.py", "old_str": "zzz\nyyy\nxxx\n", "new_str": "q"}, ctx)
    assert not res.success
    assert "old_str not found" in res.error
