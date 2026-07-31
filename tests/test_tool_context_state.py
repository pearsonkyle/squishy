"""Tests for ToolContext state management and persistence."""

from __future__ import annotations

from squishy.tools.base import ToolContext




class TestToolContextFilesRead:
    """Test files_read dictionary behavior."""

    def test_add_entries(self, tmp_path):
        """Verify entries can be added to files_read."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.files_read["test.py"] = "print('hello')\n"
        assert "test.py" in ctx.files_read
        assert ctx.files_read["test.py"] == "print('hello')\n"

    def test_overwrite_entries(self, tmp_path):
        """Verify entries can be overwritten."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.files_read["test.py"] = "version1\n"
        ctx.files_read["test.py"] = "version2\n"
        assert ctx.files_read["test.py"] == "version2\n"

    def test_remove_entries(self, tmp_path):
        """Verify entries can be removed."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        ctx.files_read["test.py"] = "content\n"
        del ctx.files_read["test.py"]
        assert "test.py" not in ctx.files_read


class TestToolContextFilesReadMeta:
    """Test files_read_meta dictionary behavior."""

    def test_cache_key_format(self, tmp_path):
        """Verify cache keys are tuples (path, offset, limit)."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        # Example cache key
        cache_key = ("test.py", 0, None)
        ctx.files_read_meta[cache_key] = {
            "content": "test",
            "total_lines": 10,
            "returned_lines": 5,
        }
        assert cache_key in ctx.files_read_meta

    def test_multiple_cache_entries(self, tmp_path):
        """Verify multiple cache entries for same file with different offsets."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        # Same file, different offsets
        ctx.files_read_meta[("test.py", 0, None)] = {"total_lines": 10}
        ctx.files_read_meta[("test.py", 5, None)] = {"total_lines": 5}
        ctx.files_read_meta[("test.py", 0, 3)] = {"total_lines": 3}

        assert len(ctx.files_read_meta) == 3

    def test_cache_invalidation_by_path(self, tmp_path):
        """Verify cache can be invalidated by path."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        # Add some cache entries
        ctx.files_read_meta[("test.py", 0, None)] = {"content": "a"}
        ctx.files_read_meta[("other.py", 0, None)] = {"content": "b"}

        # Invalidate test.py
        for key in [k for k in ctx.files_read_meta if k[0] == "test.py"]:
            del ctx.files_read_meta[key]

        assert ("test.py", 0, None) not in ctx.files_read_meta
        assert ("other.py", 0, None) in ctx.files_read_meta








class TestToolContextSandbox:
    """Test sandbox configuration."""

    def test_sandbox_enabled(self, tmp_path):
        """Verify use_sandbox can be enabled."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=True,
        )
        assert ctx.use_sandbox is True
        assert ctx.sandbox_image == "python:3.11-slim"

    def test_sandbox_disabled(self, tmp_path):
        """Verify use_sandbox can be disabled."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=False,
        )
        assert ctx.use_sandbox is False

    def test_custom_sandbox_image(self, tmp_path):
        """Verify custom sandbox image can be set."""
        ctx = ToolContext(
            working_dir=str(tmp_path),
            permission_mode="edits",
            use_sandbox=True,
            sandbox_image="custom:latest",
        )
        assert ctx.sandbox_image == "custom:latest"


