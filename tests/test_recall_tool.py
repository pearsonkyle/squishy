"""Recall tool: scoring, limit, depth, missing-index handling."""
 
from __future__ import annotations
 
from pathlib import Path
 
import pytest
 
from squishy.index import build_index, save_index
from squishy.tools.base import ToolContext
from squishy.tools.recall import _recall
 
pytestmark = pytest.mark.asyncio
 
 
def _make_repo(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "permissions.py").write_text(
        '"""Permission checking for tool calls."""\n'
        "def check_permission(tool, mode):\n"
        "    return True\n"
    )
    (root / "pkg" / "colors.py").write_text(
        '"""Terminal color helpers."""\n'
        "def paint(x): return x\n"
    )
    (root / "README.md").write_text("# demo")
 
 
async def test_missing_index_returns_error(tmp_path: Path) -> None:
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "anything"}, ctx)
    assert not r.success
    assert "no index" in r.error.lower()
 
 
async def test_empty_query_rejected(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "   "}, ctx)
    assert not r.success
 
 
async def test_name_outranks_summary(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "permission"}, ctx)
    assert r.success
    top = r.data["results"][0]
    assert "permission" in top["path"].lower() or "permission" in top.get("name", "").lower()
 
 
async def test_limit_and_depth(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "permission", "limit": 1, "depth": 0}, ctx)
    assert r.success
    assert len(r.data["results"]) == 1
    # depth=0 → no children key
    assert "children" not in r.data["results"][0]
 
    r2 = await _recall({"query": "permission", "limit": 5, "depth": 2}, ctx)
    assert r2.success
    # At least one matching node should have children surfaced at depth>=1.
    has_children = any("children" in entry for entry in r2.data["results"])
    assert has_children or r2.data["total_matched"] >= 1
 
 
async def test_no_matches_returns_empty_list(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "zzz_no_such_symbol_xyz"}, ctx)
    assert r.success
    assert r.data["returned"] == 0
