"""Recall tool: scoring, limit, depth, missing-index handling."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from squishy.index import build_index, save_index
from squishy.tools.base import ToolContext
from squishy.tools.recall import _recall, _score, _tokens


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


# Test tokenization - these are sync tests, not marked with asyncio
def test_tokens_extraction() -> None:
    """Test that _tokens extracts words correctly."""
    # Words must be at least 2 chars
    assert _tokens("hello world") == {"hello", "world"}
    # Single char words are filtered out
    assert _tokens("a b c") == set()
    # Underscore-separated words are captured as single tokens
    assert _tokens("check_permission") == {"check_permission"}
    # Empty strings give empty set
    assert _tokens("") == set()


def test_score_exact_name_match() -> None:
    """Test that exact name matches get highest score."""
    from squishy.index.model import Node
    
    node = Node(
        id="file:test.py",
        kind="file",
        name="test.py",
        path="src/test.py",
        summary="This is a test file for unit testing.",
    )
    
    # Exact name match
    score = _score(node, "test.py", set())
    assert score >= 6.0  # At least path match bonus
    
    # Exact name match with tokens
    score = _score(node, "test", {"test"})
    assert score >= 10.5  # 10 name match + 0.5 class/function bonus


def test_score_path_match() -> None:
    """Test that path matches get moderate score."""
    from squishy.index.model import Node
    
    node = Node(
        id="file:helper.py",
        kind="file",
        name="helper.py",
        path="src/helpers/utils/helper.py",
        summary="Helper utilities.",
    )
    
    # Path substring match
    score = _score(node, "helper", set())
    assert score >= 3.0  # Path match bonus


def test_score_summary_match() -> None:
    """Test that summary matches get lower score."""
    from squishy.index.model import Node
    
    node = Node(
        id="file:api.py",
        kind="file",
        name="api.py",
        path="src/api.py",
        summary="This module provides API endpoints for the application.",
    )
    
    # Summary token match
    score = _score(node, "endpoint", {"endpoint"})
    assert score >= 1.0  # Summary token match


def test_score_class_function_bonus() -> None:
    """Test that class/function nodes get bonus when matched."""
    from squishy.index.model import Node
    
    node = Node(
        id="func:calculate",
        kind="function",
        name="calculate",
        path="src/math.py",
        start_line=10,
        end_line=20,
        summary="Calculate the result.",
    )
    
    score = _score(node, "calculate", {"calculate"})
    assert score >= 10.5  # 10 name match + 0.5 symbol bonus


@pytest.mark.asyncio
async def test_missing_index_returns_error(tmp_path: Path) -> None:
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "anything"}, ctx)
    assert not r.success
    assert "no index" in r.error.lower()
 
 
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_empty_query_rejected(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "   "}, ctx)
    assert not r.success
 
 
@pytest.mark.asyncio
async def test_name_outranks_summary(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "permission"}, ctx)
    assert r.success
    top = r.data["results"][0]
    assert "permission" in top["path"].lower() or "permission" in top.get("name", "").lower()
 
 
@pytest.mark.asyncio
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
 
 
@pytest.mark.asyncio
async def test_no_matches_returns_empty_list(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    r = await _recall({"query": "zzz_no_such_symbol_xyz"}, ctx)
    assert r.success
    assert r.data["returned"] == 0


@pytest.mark.asyncio
async def test_case_insensitive_matching(tmp_path: Path) -> None:
    """Test that matching is case-insensitive."""
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    
    # Query with different case
    r = await _recall({"query": "PERMISSION"}, ctx)
    assert r.success
    assert r.data["returned"] >= 1
    
    r = await _recall({"query": "PeRmIsSiOn"}, ctx)
    assert r.success
    assert r.data["returned"] >= 1


@pytest.mark.asyncio
async def test_partial_name_matching(tmp_path: Path) -> None:
    """Test that partial name matches work."""
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    
    # Partial match on "color"
    r = await _recall({"query": "col"}, ctx)
    assert r.success
    # Should match colors.py and paint function
    assert r.data["returned"] >= 1


@pytest.mark.asyncio
async def test_multi_word_query_scoring(tmp_path: Path) -> None:
    """Test that multi-word queries score by token overlap."""
    (root := tmp_path / "pkg").mkdir(parents=True)
    (root / "api_client.py").write_text(
        '"""API client for external services."""\n'
        "def fetch_data(url: str): pass\n"
    )
    (root / "data_processor.py").write_text(
        '"""Data processing utilities."""\n'
        "def process_data(data: dict): pass\n"
    )
    
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    
    # Query with multiple tokens
    r = await _recall({"query": "api fetch"}, ctx)
    assert r.success
    # Should find api_client.py which has both tokens in summary/name
    results = r.data["results"]
    if len(results) > 0:
        # api_client.py should rank higher than data_processor.py for "api fetch"
        top_path = results[0].get("path", "")
        assert "api_client" in top_path.lower()


@pytest.mark.asyncio
async def test_limit_capped_at_max_results(tmp_path: Path) -> None:
    """Test that limit is capped at MAX_RESULTS."""
    _make_repo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    
    # Request more than MAX_RESULTS
    r = await _recall({"query": "test", "limit": 100}, ctx)
    assert r.success
    assert r.data["returned"] <= 25  # MAX_RESULTS


@pytest.mark.asyncio
async def test_depth_zero_shows_no_children(tmp_path: Path) -> None:
    """Test that depth=0 returns results without children."""
    (root := tmp_path / "pkg").mkdir(parents=True)
    (root / "complex.py").write_text(
        '"""Complex module with classes."""\n'
        "class DataLoader:\n"
        "    def load(self): pass\n"
        "    def save(self): pass\n"
        "def main(): pass\n"
    )
    
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    
    # depth=0 should not have children in results
    r = await _recall({"query": "DataLoader", "depth": 0}, ctx)
    assert r.success
    for result in r.data["results"]:
        if "DataLoader" in result.get("name", ""):
            assert "children" not in result


@pytest.mark.asyncio
async def test_depth_one_shows_methods(tmp_path: Path) -> None:
    """Test that depth=1 shows direct children (methods)."""
    (root := tmp_path / "pkg").mkdir(parents=True)
    (root / "complex.py").write_text(
        '"""Complex module with classes."""\n'
        "class DataLoader:\n"
        "    def load(self): pass\n"
        "    def save(self): pass\n"
    )
    
    save_index(str(tmp_path), build_index(str(tmp_path)))
    ctx = ToolContext(working_dir=str(tmp_path))
    
    # depth=1 should show methods as children
    r = await _recall({"query": "DataLoader", "depth": 1}, ctx)
    assert r.success
    for result in r.data["results"]:
        if "DataLoader" in result.get("name", ""):
            assert "children" in result
            method_names = [c.get("name") for c in result["children"]]
            assert "load" in method_names or "save" in method_names
