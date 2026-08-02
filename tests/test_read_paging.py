"""Paging through a large file must not trip the repeat-read guard.

read_file's own description tells the model to page with offset/limit. The
per-path cap counted every read regardless of range, so following that
instruction on a big file got refused — the top tool failure in two
consecutive sweeps (106, then 131).
"""
from __future__ import annotations

import pytest

from squishy.tools import dispatch


@pytest.fixture
def big(tmp_path):
    (tmp_path / "big.py").write_text("".join(f"line {i}\n" for i in range(4000)))
    return tmp_path


async def test_sequential_paging_is_never_refused(ctx, big):
    ctx.working_dir = str(big)
    for offset in range(0, 4000, 200):
        res = await dispatch(
            "read_file", {"path": "big.py", "offset": offset, "limit": 200}, ctx)
        assert res.success, f"refused at offset {offset}: {res.error}"


async def test_circling_back_over_the_same_region_is_still_refused(ctx, big):
    """The actual pathology: re-reading content you already have."""
    ctx.working_dir = str(big)
    last = None
    for offset in (0, 5, 10, 15, 20, 25):
        last = await dispatch(
            "read_file", {"path": "big.py", "offset": offset, "limit": 200}, ctx)
    assert not last.success
    assert "already read these lines" in last.error


async def test_the_refusal_points_somewhere_useful(ctx, big):
    ctx.working_dir = str(big)
    last = None
    for offset in (0, 2, 4, 6, 8, 10):
        last = await dispatch(
            "read_file", {"path": "big.py", "offset": offset, "limit": 500}, ctx)
    assert not last.success
    # Must suggest the way forward, not just say no.
    assert "offset" in last.error and "edit_file" in last.error


async def test_reading_a_distant_region_after_a_refusal_works(ctx, big):
    ctx.working_dir = str(big)
    for offset in (0, 1, 2, 3, 4, 5):
        await dispatch("read_file", {"path": "big.py", "offset": offset, "limit": 100}, ctx)
    res = await dispatch(
        "read_file", {"path": "big.py", "offset": 3000, "limit": 100}, ctx)
    assert res.success, res.error
