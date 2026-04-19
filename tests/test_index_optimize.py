"""Index optimization: pruning, compaction, and staleness detection."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from squishy.index import (
    _build_index_async,
    build_index,
    describe_deep_staleness,
    get_index_size,
    needs_pruning,
    optimize_for_size,
    prune_empty_dirs,
    prune_stale_summaries,
)
from squishy.index.store import save_index


def _make_repo(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text('"""A module."""\ndef alpha(): return 1\n')
    (root / "pkg" / "b.py").write_text('"""B module."""\nclass Widget: pass\n')
    (root / "README.md").write_text("# demo")


@pytest.mark.asyncio
async def test_describe_deep_staleness_fresh() -> None:
    """Fresh index should report no staleness."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_repo(root)
        idx = await _build_index_async(str(root))
        save_index(str(root), idx)

        stale_info = describe_deep_staleness(str(root))
        assert not stale_info["stale"]
        assert stale_info["reason"] is None


@pytest.mark.asyncio
async def test_describe_deep_staleness_changed_file() -> None:
    """Changed file should be detected."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_repo(root)
        idx = await _build_index_async(str(root))
        save_index(str(root), idx)

        # Modify a file and explicitly update mtime
        (root / "pkg" / "a.py").write_text('"""Modified."""\ndef alpha(): return 2\n')
        # Force mtime to be in the future relative to index generation
        new_time = time.time() + 10
        os.utime(root / "pkg" / "a.py", (new_time, new_time))

        stale_info = describe_deep_staleness(str(root))
        assert stale_info["stale"]
        assert "changed" in (stale_info.get("reason") or "")


@pytest.mark.asyncio
async def test_describe_deep_staleness_new_file() -> None:
    """New file should be detected."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_repo(root)
        idx = await _build_index_async(str(root))
        save_index(str(root), idx)

        # Add a new file
        (root / "pkg" / "c.py").write_text("def new_func(): pass\n")

        stale_info = describe_deep_staleness(str(root))
        assert stale_info["stale"]
        assert "new" in (stale_info.get("reason") or "")


def test_get_index_size() -> None:
    """Index size metrics should be computed."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_repo(root)
        idx = build_index(str(root))

        size = get_index_size(idx)
        assert size["files"] == 3
        assert size["symbols"] >= 2


def test_needs_pruning() -> None:
    """Small indexes should not need pruning."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_repo(root)
        idx = build_index(str(root))

        assert not needs_pruning(idx, max_chars=50_000)


def test_optimize_for_size() -> None:
    """Optimization should reduce index size when needed."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_repo(root)
        idx = build_index(str(root))

        stats = optimize_for_size(idx, max_chars=50_000)
        assert isinstance(stats["pruned_summaries"], int)
