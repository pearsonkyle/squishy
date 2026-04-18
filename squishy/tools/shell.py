"""Async shell execution tool. Docker sandbox when available, subprocess otherwise."""
 
from __future__ import annotations
 
import asyncio
import os
import shutil
from typing import Any
 
from squishy.tools.base import Tool, ToolContext, ToolResult
 
DEFAULT_TIMEOUT = 60
OUTPUT_CAP_STDOUT = 8000
OUTPUT_CAP_STDERR = 4000
 
 
def _docker_available() -> bool:
    return shutil.which("docker") is not None
 
 
async def _run_command(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str):
        return ToolResult(False, error="`command` is required (string)")
    timeout = float(args.get("timeout") or DEFAULT_TIMEOUT)
    cwd = args.get("cwd") or ctx.working_dir
 
    sandboxed = ctx.use_sandbox and _docker_available()

    if sandboxed:
        exec_args = [
            "docker", "run", "--rm",
            "-v", f"{ctx.working_dir}:/work",
            "-w", "/work",
            "--network=none",
            ctx.sandbox_image,
            "sh", "-c", command,
        ]
        exec_cwd = ctx.working_dir
        exec_env = os.environ.copy()
    else:
        exec_args = ["sh", "-c", command]
        exec_cwd = cwd
        exec_env = None  # inherit parent environment naturally

    try:
        proc = await asyncio.create_subprocess_exec(
            *exec_args,
            cwd=exec_cwd,
            env=exec_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        return ToolResult(False, error=str(e))

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ToolResult(False, error=f"command timed out after {timeout}s")

    hint = ""
    if sandboxed and proc.returncode == 127:
        hint = (
            " (command not found in sandbox — "
            "try --no-sandbox or install the tool in the sandbox image)"
        )

    stderr_cap = OUTPUT_CAP_STDERR - len(hint)
    stderr_text = stderr_b.decode("utf-8", errors="replace")[-stderr_cap:] + hint

    return ToolResult(
        True,
        data={
            "command": command,
            "exit_code": proc.returncode,
            "stdout": stdout_b.decode("utf-8", errors="replace")[-OUTPUT_CAP_STDOUT:],
            "stderr": stderr_text,
            "sandboxed": sandboxed,
        },
        # Escape brackets so Rich does not swallow the sandbox tag
        display=f"exit={proc.returncode}" + (" \\[sandbox]" if sandboxed else ""),
    )
 
 
run_command = Tool(
    name="run_command",
    description="Run a shell command and capture stdout/stderr/exit code. "
                "Sandboxed in Docker when available.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "default": DEFAULT_TIMEOUT},
            "cwd": {"type": "string"},
        },
        "required": ["command"],
    },
    run=_run_command,
    mutates=True,
)
 
SHELL_TOOLS: list[Tool] = [run_command]