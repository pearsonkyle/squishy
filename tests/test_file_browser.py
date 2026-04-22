"""Tests for the file browser feature (@filename syntax)."""

from __future__ import annotations

import os
import tempfile

import pytest

from squishy.file_browser import (
    FILE_PATTERN,
    FileReference,
    format_reference_list,
    inject_references,
    parse_references,
    wrap_file_content,
)


def test_parse_single_reference():
    """Test parsing a single @filename reference."""
    text = "Read @app.py for context"
    with tempfile.TemporaryDirectory() as tmpdir:
        app_py = os.path.join(tmpdir, "app.py")
        with open(app_py, "w") as f:
            f.write("print('hello')")

        refs = parse_references(text, tmpdir)
        assert len(refs) == 1
        assert refs[0].path == "app.py"
        assert "print('hello')" in refs[0].content


def test_parse_multiple_references():
    """Test parsing multiple @filename references."""
    text = "Check @a.py and @b.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        a_py = os.path.join(tmpdir, "a.py")
        b_py = os.path.join(tmpdir, "b.py")
        with open(a_py, "w") as f:
            f.write("a = 1")
        with open(b_py, "w") as f:
            f.write("b = 2")

        refs = parse_references(text, tmpdir)
        assert len(refs) == 2
        paths = {r.path for r in refs}
        assert paths == {"a.py", "b.py"}


def test_parse_nonexistent_file():
    """Test that nonexistent files don't cause errors."""
    text = "Read @nonexistent.py"
    refs = parse_references(text, "/tmp")
    assert len(refs) == 0


def test_parse_relative_path():
    """Test resolving relative paths."""
    text = "Check @src/utils.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "src")
        os.makedirs(src_dir)
        utils_py = os.path.join(src_dir, "utils.py")
        with open(utils_py, "w") as f:
            f.write("def helper(): pass")

        refs = parse_references(text, tmpdir)
        assert len(refs) == 1
        assert refs[0].absolute_path == utils_py


def test_inject_references():
    """Test replacing @filename with wrapped content."""
    text = "Fix the bug in @app.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        app_py = os.path.join(tmpdir, "app.py")
        with open(app_py, "w") as f:
            f.write("def bad_func():\n  return")

        result, refs = inject_references(text, tmpdir)
        assert len(refs) == 1
        assert "@app.py" not in result
        assert 'path="app.py"' in result
        assert "def bad_func():" in result


def test_inject_preserves_surrounding_text():
    """Test that surrounding text is preserved."""
    text = "Please @read this file and fix it"
    with tempfile.TemporaryDirectory() as tmpdir:
        read_py = os.path.join(tmpdir, "read")
        with open(read_py, "w") as f:
            f.write("content")

        result, _ = inject_references(text, tmpdir)
        assert "Please" in result
        assert "and fix it" in result


def test_wrap_file_content():
    """Test wrapping file content with metadata."""
    content = "line1\nline2\nline3"
    wrapped = wrap_file_content(content)
    
    assert '<file path="" total_lines="3">' in wrapped
    assert "line1" in wrapped
    assert "line2" in wrapped
    assert "line3" in wrapped


def test_wrap_file_content_truncation():
    """Test file content truncation."""
    content = "\n".join(f"line{i}" for i in range(150))
    wrapped = wrap_file_content(content, max_lines=100)
    
    assert "line99" in wrapped
    assert "[Note: 50 lines truncated]" in wrapped


def test_format_reference_list():
    """Test formatting reference list for display."""
    refs = [
        FileReference("app.py", "/app.py", "content\ncontent"),
        FileReference("utils.py", "/utils.py", "x = 1"),
    ]
    
    formatted = format_reference_list(refs)
    assert "Referenced files:" in formatted
    assert "app.py (2 lines)" in formatted
    assert "utils.py (1 lines)" in formatted


def test_empty_input():
    """Test handling of empty input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result, refs = inject_references("", tmpdir)
        assert result == ""
        assert refs == []


def test_no_references():
    """Test input without any @references."""
    text = "Hello world, no files here"
    with tempfile.TemporaryDirectory() as tmpdir:
        result, refs = inject_references(text, tmpdir)
        assert result == text
        assert refs == []


def test_pattern_matches_various_formats():
    """Test that the pattern matches different path formats."""
    test_cases = [
        "file.py",
        "dir/file.py",
        "./relative.py",
        "/absolute/path.py",
    ]
    
    for case in test_cases:
        match = FILE_PATTERN.search(f"@{case}")
        assert match is not None, f"Failed to match @{case}"
        assert match.group(1) == case
