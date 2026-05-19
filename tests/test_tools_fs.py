from __future__ import annotations


from squishy.tools.fs import (
    edit_file,
    list_directory,
    read_file,
    search_files,
    undo_edit,
    write_file,
)



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


async def test_edit_file_unescape_quotes(ctx):
    """edit_file should auto-unescape over-escaped quotes in old_str/new_str."""
    await write_file.run(
        {"path": "docstr.py", "content": '"""Module docstring."""\nx = 1\n'}, ctx
    )
    # Model sends escaped quotes (\") that don't match actual quotes (")
    r = await edit_file.run(
        {
            "path": "docstr.py",
            "old_str": '\\"\\"\\"Module docstring.\\"\\"\\"',
            "new_str": '\\"\\"\\"Updated docstring.\\"\\"\\"',
        },
        ctx,
    )
    assert r.success, r.error
    assert "escape sequences normalized" in r.data.get("note", "")
    content = (ctx.working_dir / "docstr.py").read_text() if hasattr(ctx.working_dir, "read_text") else open(ctx.working_dir + "/docstr.py").read()
    assert '"""Updated docstring."""' in content


async def test_edit_file_unescape_newlines(ctx):
    """edit_file should auto-unescape literal \\n in new_str."""
    await write_file.run(
        {"path": "lines.py", "content": "line1\nline2\n"}, ctx
    )
    r = await edit_file.run(
        {
            "path": "lines.py",
            "old_str": "line1\\nline2",
            "new_str": "line1\\nline2\\nline3",
        },
        ctx,
    )
    assert r.success, r.error
    content = open(ctx.working_dir + "/lines.py").read() if isinstance(ctx.working_dir, str) else (ctx.working_dir / "lines.py").read_text()
    assert "line3" in content


async def test_edit_file_unescape_not_applied_when_exact_match(ctx):
    """Unescape should not fire when old_str already matches exactly."""
    await write_file.run(
        {"path": "esc.py", "content": 'path = "hello\\nworld"\n'}, ctx
    )
    # old_str with literal backslash-n should match the file exactly
    r = await edit_file.run(
        {
            "path": "esc.py",
            "old_str": 'path = "hello\\nworld"',
            "new_str": 'path = "hello\\nworld\\n!"',
        },
        ctx,
    )
    assert r.success, r.error
    # Should not have "escape sequences normalized" since exact match worked
    assert r.data.get("note") is None


async def test_edit_file_unescape_preserves_intentional_backslashes_in_new_str(ctx):
    """When old_str is over-escaped but new_str has no escapes, new_str
    should be written verbatim — not mangled by _unescape_str."""
    await write_file.run(
        {"path": "regex.py", "content": 'pattern = "hello"\n'}, ctx
    )
    # Model over-escapes old_str (sends \" for "), but new_str is plain text
    # with no backslash sequences — it must NOT be unescaped.
    r = await edit_file.run(
        {
            "path": "regex.py",
            "old_str": 'pattern = \\"hello\\"',
            "new_str": 'pattern = "world"',
        },
        ctx,
    )
    assert r.success, r.error
    content = open(ctx.working_dir + "/regex.py").read() if isinstance(ctx.working_dir, str) else (ctx.working_dir / "regex.py").read_text()
    assert 'pattern = "world"' in content


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


async def test_undo_edit_reverts_last_change(ctx):
    """undo_edit should restore the file to its pre-edit content."""
    await write_file.run({"path": "undo_test.py", "content": "original\n"}, ctx)
    r = await edit_file.run(
        {"path": "undo_test.py", "old_str": "original", "new_str": "modified"}, ctx
    )
    assert r.success

    r = await undo_edit.run({}, ctx)
    assert r.success
    assert "undo_test.py" in r.data["path"]

    # read_file strips trailing newlines via splitlines()+join
    r = await read_file.run({"path": "undo_test.py"}, ctx)
    assert r.success
    assert r.data["content"] == "original"


async def test_undo_edit_empty_stack(ctx):
    """undo_edit with no prior edits should fail gracefully."""
    r = await undo_edit.run({}, ctx)
    assert not r.success
    assert "Nothing to undo" in r.error


async def test_undo_edit_multiple(ctx):
    """Multiple undo calls should revert edits in LIFO order."""
    await write_file.run({"path": "multi.py", "content": "v1\n"}, ctx)
    await edit_file.run({"path": "multi.py", "old_str": "v1", "new_str": "v2"}, ctx)
    await edit_file.run({"path": "multi.py", "old_str": "v2", "new_str": "v3"}, ctx)

    # Undo second edit: v3 -> v2
    r = await undo_edit.run({}, ctx)
    assert r.success
    r = await read_file.run({"path": "multi.py"}, ctx)
    assert r.data["content"] == "v2"

    # Undo first edit: v2 -> v1
    r = await undo_edit.run({}, ctx)
    assert r.success
    r = await read_file.run({"path": "multi.py"}, ctx)
    assert r.data["content"] == "v1"


async def test_edit_file_reports_unrecognized_keys(ctx):
    """v6e: when required params are missing AND unknown params are present,
    the error message should list the silently-ignored keys so the model
    can correct its mental model on the next turn (react-datepicker-4282
    pattern: model sent {start_line, end_line, offset, text} repeatedly)."""
    r = await edit_file.run(
        {
            "path": "x.py",
            "start_line": 1,
            "end_line": 5,
            "offset": 0,
            "text": "...",
        },
        ctx,
    )
    assert not r.success
    assert "Missing or non-string parameter(s)" in r.error
    assert "Unrecognized parameters (silently ignored)" in r.error
    # All four foreign keys should appear so the model knows exactly what
    # was dropped.
    for k in ("start_line", "end_line", "offset", "text"):
        assert k in r.error


async def test_edit_file_no_unrecognized_hint_when_only_aliases(ctx):
    """v6e: the unrecognized-keys hint should not fire when the caller used
    a recognized alias (e.g. file_path, original) but happened to omit a
    required param.  Avoids false positives on legitimate alias usage."""
    r = await edit_file.run(
        {
            "file_path": "x.py",
            "original": "foo",
            # missing new_str / new_string / etc.
        },
        ctx,
    )
    assert not r.success
    assert "Missing or non-string parameter(s)" in r.error
    assert "new_str" in r.error
    # No unknown-keys clause should appear.
    assert "Unrecognized parameters" not in r.error
