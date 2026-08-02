"""`not a directory` should say what to do instead."""
from __future__ import annotations

from squishy.tools import dispatch


async def test_pointing_list_directory_at_a_file(ctx, tmp_path):
    (tmp_path / "printer.ts").write_text("export const x = 1;\n")
    ctx.working_dir = str(tmp_path)
    res = await dispatch("list_directory", {"path": "printer.ts"}, ctx)
    assert not res.success
    assert "is a file" in res.error
    assert "read_file" in res.error


async def test_dropped_extension_surfaces_the_real_file(ctx, tmp_path):
    """`src/printer` for `src/printer.ts` — seen twice in the sweep.

    Either recovery route is fine (naming the candidate, or listing the
    parent); what matters is that the real filename reaches the model.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "printer.ts").write_text("x\n")
    ctx.working_dir = str(tmp_path)
    res = await dispatch("list_directory", {"path": "src/printer"}, ctx)
    assert not res.success
    assert "printer.ts" in res.error


async def test_unknown_dir_lists_its_parent(ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("x\n")
    ctx.working_dir = str(tmp_path)
    res = await dispatch("list_directory", {"path": "src/nope"}, ctx)
    assert not res.success
    assert "alpha.py" in res.error


async def test_a_real_directory_still_lists(ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "alpha.py").write_text("x\n")
    ctx.working_dir = str(tmp_path)
    res = await dispatch("list_directory", {"path": "src"}, ctx)
    assert res.success, res.error
