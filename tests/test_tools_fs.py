from __future__ import annotations

import pytest

from squishy.tools.fs import (
    edit_file,
    list_directory,
    read_file,
    search_files,
    write_file,
)

pytestmark = pytest.mark.asyncio


async def test_write_and_read_roundtrip(ctx):
    r = await write_file.run({"path": "hello.py", "content": "print('hi')\n"}, ctx)
    assert r.success, r.error

    r = await read_file.run({"path": "hello.py"}, ctx)
    assert r.success
    assert "print('hi')" in r.data["content"]
    assert ctx.files_read["hello.py"].startswith("print")


async def test_write_file_rejects_any_existing_file(ctx, tmp_path):
    (tmp_path / "small.py").write_text("x = 1\n")

    r = await write_file.run({"path": "small.py", "content": "y = 2\n"}, ctx)
    assert not r.success
    assert "already exists" in r.error
    assert "edit_file" in r.error


async def test_edit_file_unique_match(ctx):
    await write_file.run({"path": "app.py", "content": "def a():\n    pass\n"}, ctx)
    r = await edit_file.run({"path": "app.py", "old_str": "def a()", "new_str": "def b()"}, ctx)
    assert r.success
    assert r.data["replacements"] == 1

    r = await read_file.run({"path": "app.py"}, ctx)
    assert "def b()" in r.data["content"]
    assert "def a()" not in r.data["content"]


async def test_edit_file_ambiguous_match_rejected_without_replace_all(ctx):
    await write_file.run({"path": "app.py", "content": "x = 1\nx = 2\n"}, ctx)
    r = await edit_file.run({"path": "app.py", "old_str": "x", "new_str": "y"}, ctx)
    assert not r.success
    assert "matches 2 times" in r.error


async def test_edit_file_replace_all(ctx):
    await write_file.run({"path": "app.py", "content": "x = 1\nx = 2\n"}, ctx)
    r = await edit_file.run(
        {"path": "app.py", "old_str": "x", "new_str": "y", "replace_all": True}, ctx
    )
    assert r.success
    assert r.data["replacements"] == 2


async def test_edit_file_missing_match(ctx):
    await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)
    r = await edit_file.run({"path": "app.py", "old_str": "nope", "new_str": "yo"}, ctx)
    assert not r.success
    assert "not found" in r.error


async def test_list_directory_hides_dotfiles(ctx, tmp_path):
    (tmp_path / "visible.txt").write_text("")
    (tmp_path / ".hidden").write_text("")
    (tmp_path / ".git").mkdir()
    r = await list_directory.run({"path": "."}, ctx)
    assert r.success
    names = [e["name"] for e in r.data["entries"]]
    assert "visible.txt" in names
    assert ".hidden" not in names
    assert ".git" not in names


async def test_search_files(ctx, tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    (tmp_path / "b.py").write_text("def bar():\n    pass\n")
    r = await search_files.run({"pattern": r"def ", "path": "."}, ctx)
    assert r.success
    assert r.data["count"] >= 2


async def test_read_file_offset_limit(ctx, tmp_path):
    (tmp_path / "x.txt").write_text("\n".join(str(i) for i in range(20)))
    r = await read_file.run({"path": "x.txt", "offset": 5, "limit": 3}, ctx)
    assert r.success
    assert r.data["content"] == "5\n6\n7"


async def test_read_file_dedup_returns_cache_hit(ctx, tmp_path):
    """Re-reading the same window returns a cache_hit marker so the LLM
    realizes it has already seen this file and should use what it has.
    """
    (tmp_path / "dup.py").write_text("a\nb\nc\n")
    r1 = await read_file.run({"path": "dup.py"}, ctx)
    assert r1.success
    assert not r1.data.get("cache_hit")

    r2 = await read_file.run({"path": "dup.py"}, ctx)
    assert r2.success
    assert r2.data.get("cache_hit") is True
    assert "already read" in r2.data.get("note", "")
    assert r2.data["content"] == r1.data["content"]


async def test_read_file_different_window_misses_cache(ctx, tmp_path):
    (tmp_path / "dup.py").write_text("\n".join(str(i) for i in range(10)))
    r1 = await read_file.run({"path": "dup.py", "offset": 0, "limit": 3}, ctx)
    assert r1.success
    assert not r1.data.get("cache_hit")

    # Different offset/limit -> fresh read, not a cache hit.
    r2 = await read_file.run({"path": "dup.py", "offset": 5, "limit": 3}, ctx)
    assert r2.success
    assert not r2.data.get("cache_hit")
    assert r2.data["content"] == "5\n6\n7"


async def test_mutating_file_invalidates_read_cache(ctx):
    await write_file.run({"path": "v.py", "content": "one\n"}, ctx)
    r1 = await read_file.run({"path": "v.py"}, ctx)
    assert r1.data["content"] == "one"

    # Use edit_file to change content (write_file refuses existing files).
    await edit_file.run({"path": "v.py", "old_str": "one", "new_str": "two"}, ctx)
    r2 = await read_file.run({"path": "v.py"}, ctx)
    assert not r2.data.get("cache_hit")
    assert r2.data["content"] == "two"


async def test_edit_file_invalidates_read_cache(ctx):
    await write_file.run({"path": "v.py", "content": "alpha\n"}, ctx)
    r1 = await read_file.run({"path": "v.py"}, ctx)
    assert r1.data["content"] == "alpha"

    await edit_file.run({"path": "v.py", "old_str": "alpha", "new_str": "beta"}, ctx)
    r2 = await read_file.run({"path": "v.py"}, ctx)
    assert not r2.data.get("cache_hit")
    assert r2.data["content"] == "beta"


async def test_edit_file_trailing_whitespace_fuzzy_match(ctx):
    """When old_str differs only in trailing whitespace, edit should still succeed."""
    await write_file.run(
        {"path": "ws.py", "content": "def foo():  \n    pass\n"}, ctx
    )
    # old_str lacks the trailing spaces on line 1
    r = await edit_file.run(
        {"path": "ws.py", "old_str": "def foo():\n    pass", "new_str": "def bar():\n    pass"},
        ctx,
    )
    assert r.success
    assert "trailing whitespace normalized" in (r.data.get("note", "") + (r.display or ""))


async def test_edit_file_diagnostic_hint_on_miss(ctx):
    """When old_str not found, error should hint at the right line if the first
    line exists but indentation differs."""
    await write_file.run(
        {"path": "hint.py", "content": "    def foo():\n        pass\n"}, ctx
    )
    # old_str has extra lines that don't match, but first line content exists
    r = await edit_file.run(
        {"path": "hint.py", "old_str": "def foo():\n    return 1", "new_str": "def bar():\n    return 2"},
        ctx,
    )
    assert not r.success
    assert "appears at line" in r.error
