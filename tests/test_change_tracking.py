"""Tests for change tracking: file operations and plan state persistence."""

from __future__ import annotations

from conftest import FakeClient

from squishy.agent import Agent
from squishy.client import CompletionResult, ToolCall
from squishy.config import Config
from squishy.display import Display


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


async def test_file_operations_tracking_created(ctx):
    """Verify files_read dict is populated for read operations after write."""
    from squishy.tools.fs import read_file, write_file

    result = await write_file.run(
        {"path": "new.py", "content": "# New file\n"}, ctx
    )
    assert result.success

    # Read the file to populate files_read
    r = await read_file.run({"path": "new.py"}, ctx)
    assert r.success

    # Check ToolContext files_read was populated
    assert "new.py" in ctx.files_read


async def test_file_operations_tracking_edited(ctx):
    """Verify files_read dict is populated after edit operations."""
    from squishy.tools.fs import edit_file, read_file, write_file

    # Create file first
    await write_file.run({"path": "app.py", "content": "x = 1\n"}, ctx)

    # Edit it
    result = await edit_file.run(
        {"path": "app.py", "old_str": "x = 1", "new_str": "x = 2"}, ctx
    )
    assert result.success

    # Read it to populate files_read (edit invalidates cache but read populates)
    r = await read_file.run({"path": "app.py"}, ctx)
    assert r.success

    # Verify file was actually edited - it should be in files_read
    assert "app.py" in ctx.files_read






async def test_read_cache_invalidation_on_write(ctx, tmp_path):
    """Verify cache is cleared after file mutation."""
    from squishy.tools.fs import read_file, write_file

    # Write initial content
    await write_file.run({"path": "data.txt", "content": "version1\n"}, ctx)

    # Read it (populates cache)
    result = await read_file.run({"path": "data.txt"}, ctx)
    assert result.success
    # Note: cache_hit only appears on subsequent reads of same offset/limit

    # Edit content (should invalidate cache); write_file refuses existing files
    from squishy.tools.fs import edit_file
    await edit_file.run({"path": "data.txt", "old_str": "version1", "new_str": "version2"}, ctx)

    # Read again - should show new content (cache invalidated)
    result = await read_file.run({"path": "data.txt"}, ctx)
    assert result.success
    # cache_hit should be False or not present since we invalidated the cache on edit
    if "cache_hit" in result.data:
        assert not result.data["cache_hit"]
    assert "version2" in result.data["content"]


async def test_read_cache_invalidation_on_edit(ctx, tmp_path):
    """Verify cache is cleared after edit_file operation."""
    from squishy.tools.fs import edit_file, read_file, write_file

    # Create file
    await write_file.run({"path": "code.py", "content": "x = 1\n"}, ctx)

    # Read it
    result = await read_file.run({"path": "code.py"}, ctx)
    assert result.success
    original_content = result.data["content"]

    # Edit it
    await edit_file.run(
        {"path": "code.py", "old_str": "x = 1", "new_str": "y = 2"}, ctx
    )

    # Read again - should show new content (cache invalidated)
    result = await read_file.run({"path": "code.py"}, ctx)
    assert result.success
    if "cache_hit" in result.data:
        assert not result.data["cache_hit"]
    assert "y = 2" in result.data["content"]


async def test_multiple_edits_same_file_tracking(ctx, tmp_path):
    """Verify multiple edits to same file are tracked correctly."""
    from squishy.tools.fs import edit_file, read_file, write_file

    # Create initial file
    await write_file.run({"path": "app.py", "content": "a = 1\nb = 2\nc = 3\n"}, ctx)

    # Multiple edits
    await edit_file.run(
        {"path": "app.py", "old_str": "a = 1", "new_str": "A = 1"}, ctx
    )
    await edit_file.run(
        {"path": "app.py", "old_str": "b = 2", "new_str": "B = 2"}, ctx
    )
    await edit_file.run(
        {"path": "app.py", "old_str": "c = 3", "new_str": "C = 3"}, ctx
    )

    # Read file to populate files_read
    r = await read_file.run({"path": "app.py"}, ctx)
    assert r.success

    # Verify final content in files_read
    assert "app.py" in ctx.files_read
    content = ctx.files_read["app.py"]
    assert "A = 1" in content
    assert "B = 2" in content
    assert "C = 3" in content


async def test_files_created_vs_edited_tracking(tmp_path):
    """Verify files_created and files_edited are tracked separately."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 8

    script = [
        CompletionResult(
            tool_calls=[
                _tc("write_file", {"path": "new.py", "content": "# new\n"}, call_id="c1")
            ]
        ),
        CompletionResult(
            tool_calls=[
                _tc("write_file", {"path": "another.py", "content": "# another\n"}, call_id="c2")
            ]
        ),
        CompletionResult(
            tool_calls=[
                _tc("edit_file", {"path": "new.py", "old_str": "# new\n", "new_str": "# modified\n"}, call_id="c3")
            ]
        ),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Create and modify files")

    assert result.success
    assert "new.py" in result.files_created
    assert "another.py" in result.files_created
    # new.py appears in files_edited too (edited after creation)
    assert "new.py" in result.files_edited


async def test_context_files_read_accumulates(tmp_path):
    """Verify files_read dict accumulates across multiple tool calls."""
    cfg = Config()
    cfg.working_dir = str(tmp_path)
    cfg.permission_mode = "edits"
    cfg.max_turns = 6

    # Create multiple files
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text(f"# file {i}\n")

    script = [
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file0.py"}, call_id="c1")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file1.py"}, call_id="c2")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file2.py"}, call_id="c3")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file3.py"}, call_id="c4")]),
        CompletionResult(tool_calls=[_tc("read_file", {"path": "file4.py"}, call_id="c5")]),
        CompletionResult(text="Done", tool_calls=[]),
    ]

    fake = FakeClient(script=script)
    agent = Agent(cfg, fake, Display())  # type: ignore[arg-type]
    result = await agent.run("Read many files")

    assert result.success

    # All 5 files should be in files_read
    for i in range(5):
        assert f"file{i}.py" in agent.tool_ctx.files_read

    # Verify content was read
    for i in range(5):
        assert f"# file {i}" in agent.tool_ctx.files_read[f"file{i}.py"]






