from __future__ import annotations
 
import pytest
 
from squishy.tools.fs import (
    WRITE_LIMIT_LINES,
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
 
 
async def test_write_file_rejects_large_existing_file(ctx, tmp_path):
    big = tmp_path / "big.py"
    big.write_text("# line\n" * (WRITE_LIMIT_LINES + 5))
 
    r = await write_file.run({"path": "big.py", "content": "tiny"}, ctx)
    assert not r.success
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
