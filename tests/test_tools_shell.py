from __future__ import annotations
 
import pytest
 
from squishy.tools.shell import run_command
 
pytestmark = pytest.mark.asyncio
 
 
async def test_run_command_success(ctx):
    r = await run_command.run({"command": "echo hello"}, ctx)
    assert r.success
    assert r.data["exit_code"] == 0
    assert "hello" in r.data["stdout"]
 
 
async def test_run_command_nonzero_exit(ctx):
    r = await run_command.run({"command": "sh -c 'exit 7'"}, ctx)
    # Non-zero exit now reports as tool failure so a 7B model can't mistake a
    # crashing command for success, but exit_code and outputs stay in `data`.
    assert not r.success
    assert r.data["exit_code"] == 7
    assert "exited 7" in r.error


async def test_run_command_truncation_is_signalled(ctx):
    # Produce ~20KB of stdout to trip the OUTPUT_CAP_STDOUT limit.
    r = await run_command.run(
        {"command": "python -c \"print('x' * 20000)\""}, ctx
    )
    assert r.success
    assert r.data["truncated"] is True
    assert "truncated" in r.data["stdout"]
 
 
async def test_run_command_timeout(ctx):
    r = await run_command.run({"command": "sleep 5", "timeout": 1}, ctx)
    assert not r.success
    assert "timed out" in r.error
 
 
async def test_run_command_cwd(ctx, tmp_path):
    (tmp_path / "marker").write_text("here")
    r = await run_command.run({"command": "ls", "cwd": str(tmp_path)}, ctx)
    assert r.success
    assert "marker" in r.data["stdout"]
