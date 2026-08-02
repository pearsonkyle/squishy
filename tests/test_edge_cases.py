"""Edge cases and error handling tests."""

from __future__ import annotations




def _tc(name: str, args: dict, call_id: str = "c1"):
    from squishy.client import ToolCall
    return ToolCall(id=call_id, name=name, args=args)





class TestReadCacheEdgeCases:
    """Test edge cases in file read caching."""

    async def test_read_file_dedup_same_offset(self, ctx):
        """Re-reading same offset returns cache_hit marker."""
        from squishy.tools.fs import read_file, write_file

        # Create test file
        await write_file.run(
            {"path": "test.py", "content": "line1\nline2\nline3\n"}, ctx
        )

        # Read first time - no cache hit
        result1 = await read_file.run(
            {"path": "test.py", "offset": 0, "limit": 2},
            ctx,
        )
        assert result1.success
        assert "cache_hit" not in result1.data

        # Read again with same offset/limit - should be cache hit
        result2 = await read_file.run(
            {"path": "test.py", "offset": 0, "limit": 2},
            ctx,
        )
        assert result2.success
        assert result2.data.get("cache_hit") is True

    async def test_read_file_different_offset_misses_cache(self, ctx):
        """Reading different offset should miss cache."""
        from squishy.tools.fs import read_file, write_file

        # Create test file
        await write_file.run(
            {"path": "test.py", "content": "line1\nline2\nline3\n"}, ctx
        )

        # Read with offset 0
        result1 = await read_file.run(
            {"path": "test.py", "offset": 0, "limit": 2},
            ctx,
        )
        assert result1.success
        assert "cache_hit" not in result1.data

        # Read with different offset - should be cache miss
        result2 = await read_file.run(
            {"path": "test.py", "offset": 1, "limit": 2},
            ctx,
        )
        assert result2.success
        # Should not be a cache hit since offset is different
        if "cache_hit" in result2.data:
            assert not result2.data["cache_hit"]
        # Content should be different (starting from line 2)
        assert "line2" in result2.data["content"]

    async def test_read_nonexistent_file(self, ctx):
        """Reading a nonexistent file should fail gracefully."""
        from squishy.tools.fs import read_file

        result = await read_file.run({"path": "does_not_exist.py"}, ctx)
        assert not result.success
        assert "file not found" in result.error.lower()

class TestEditEdgeCases:
    """Test edge cases in file editing."""

    async def test_edit_file_no_match(self, ctx):
        """Edit with no match should fail."""
        from squishy.tools.fs import edit_file, write_file

        await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "goodbye", "new_str": "world"},
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_edit_file_ambiguous_multiple_matches(self, ctx):
        """Edit with multiple matches without replace_all should fail."""
        from squishy.tools.fs import edit_file, write_file

        await write_file.run({"path": "app.py", "content": "x\nx\nx\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "x", "new_str": "y"},
            ctx,
        )
        assert not result.success
        assert "matches" in result.error.lower()
        assert "replace_all" in result.error.lower()

    async def test_edit_file_replace_all_with_no_matches(self, ctx):
        """replace_all=True with no matches should still fail."""
        from squishy.tools.fs import edit_file, write_file

        await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "goodbye", "new_str": "world", "replace_all": True},
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_edit_file_empty_old_str(self, ctx):
        """Empty old_str should be rejected or handled."""
        from squishy.tools.fs import edit_file, write_file

        await write_file.run({"path": "app.py", "content": "hello\n"}, ctx)

        result = await edit_file.run(
            {"path": "app.py", "old_str": "", "new_str": "world"},
            ctx,
        )
        # Empty old_str is problematic - either fail or replace everything
        assert not result.success

class TestWriteEdgeCases:
    """Test edge cases in file writing."""

    async def test_write_file_empty_content(self, ctx):
        """Writing empty content should succeed."""
        from squishy.tools.fs import write_file

        result = await write_file.run(
            {"path": "empty.py", "content": ""},
            ctx,
        )
        assert result.success

    async def test_write_file_binary_like_content(self, ctx):
        """Writing content that looks like binary should work as text."""
        from squishy.tools.fs import write_file

        # Content with special characters
        content = "#!/usr/bin/env python\nprint('hello')\n"

        result = await write_file.run(
            {"path": "script.py", "content": content},
            ctx,
        )
        assert result.success

    async def test_write_file_large_content(self, ctx):
        """Writing large content should succeed."""
        from squishy.tools.fs import write_file

        # 1000 lines of content
        large_content = "\n".join(f"# line {i}" for i in range(1000)) + "\n"

        result = await write_file.run(
            {"path": "large.py", "content": large_content},
            ctx,
        )
        assert result.success





