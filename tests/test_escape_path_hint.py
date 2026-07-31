"""An out-of-tree absolute path should still point at the real file."""
from __future__ import annotations

from squishy.tools import dispatch


async def test_escape_refusal_suggests_the_real_path(ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "integer.ts").write_text("export const x = 1;\n")
    ctx.working_dir = str(tmp_path)

    res = await dispatch(
        "read_file", {"path": "/test/fast-check/src/integer.ts"}, ctx)
    assert not res.success
    # The refusal stands — this is a sandbox boundary, not a typo.
    assert "outside working directory" in res.error
    # ...but it names the file the model actually wanted.
    assert "src/integer.ts" in res.error


async def test_escape_refusal_without_a_candidate_names_the_root(ctx, tmp_path):
    ctx.working_dir = str(tmp_path)
    res = await dispatch("read_file", {"path": "/etc/passwd"}, ctx)
    assert not res.success
    assert "outside working directory" in res.error
    assert str(tmp_path) in res.error
