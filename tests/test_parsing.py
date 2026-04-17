from __future__ import annotations
 
from squishy.tools import REGISTRY, dispatch, openai_schemas
from squishy.tools.base import ToolContext
 
 
async def test_unknown_tool_returns_failure(ctx):
    r = await dispatch("does_not_exist", {}, ctx)
    assert not r.success
    assert "unknown tool" in r.error
 
 
async def test_bad_args_returns_failure(ctx):
    r = await dispatch("read_file", {}, ctx)
    assert not r.success
    assert "path" in r.error
 
 
async def test_dispatch_catches_exceptions(ctx):
    r = await dispatch("read_file", {"path": "/definitely/not/here/xyz"}, ctx)
    assert not r.success
 
 
async def test_plan_mode_blocks_mutations():
    ctx = ToolContext(working_dir=".", permission_mode="plan")
    r = await dispatch("write_file", {"path": "x", "content": "y"}, ctx)
    assert not r.success
    assert "plan mode" in r.error
 
 
async def test_edits_mode_prompts_run_command():
    # In edits mode, run_command is gated on a prompt; dispatch refuses when
    # no prompt_fn is provided (no TTY / library usage).
    ctx = ToolContext(working_dir=".", permission_mode="edits")
    r = await dispatch("run_command", {"command": "echo x"}, ctx)
    assert not r.success
    assert "approval" in r.error
 
 
async def test_edits_mode_prompt_fn_approves():
    ctx = ToolContext(working_dir=".", permission_mode="edits")
 
    async def approve(_tool, _args):
        return True
 
    r = await dispatch("run_command", {"command": "echo x"}, ctx, prompt_fn=approve)
    assert r.success
 
 
async def test_edits_mode_prompt_fn_declines():
    ctx = ToolContext(working_dir=".", permission_mode="edits")
 
    async def decline(_tool, _args):
        return False
 
    r = await dispatch("run_command", {"command": "echo x"}, ctx, prompt_fn=decline)
    assert not r.success
    assert "declined" in r.error
 
 
async def test_yolo_mode_allows_all(ctx):
    assert ctx.permission_mode == "yolo"
    r = await dispatch("list_directory", {"path": "."}, ctx)
    assert r.success
 
 
def test_openai_schemas_has_required_fields():
    schemas = openai_schemas()
    names = [s["function"]["name"] for s in schemas]
    for expected in (
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "run_command",
    ):
        assert expected in names
    for s in schemas:
        assert "parameters" in s["function"]
        assert s["function"]["parameters"]["type"] == "object"
 
 
def test_registry_is_complete():
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
        "run_command",
    ):
        assert name in REGISTRY
