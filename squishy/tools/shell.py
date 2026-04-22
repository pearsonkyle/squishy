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
# Host env vars that are safe to forward into the sandbox. Everything else
# (API keys, tokens, local paths) is dropped so a stray `env` call can't
# exfiltrate the agent's credentials.
SANDBOX_ALLOWED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TERM")


def _cap_output(raw: bytes, cap: int) -> tuple[str, bool]:
    """Return (decoded-and-capped, truncated?).

    Keeps the tail — usually the most diagnostic part of long outputs
    (tracebacks, test failures). Prepends an explicit marker when truncation
    happens so the model knows it didn't see everything.
    """
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= cap:
        return text, False
    dropped = len(text) - cap
    marker = f"…<truncated {dropped} bytes of head>\n"
    return marker + text[-cap:], True
 
 
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
        exec_env = {k: v for k, v in os.environ.items() if k in SANDBOX_ALLOWED_ENV}
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

    stdout_text, stdout_truncated = _cap_output(stdout_b, OUTPUT_CAP_STDOUT)
    stderr_text, stderr_truncated = _cap_output(stderr_b, OUTPUT_CAP_STDERR - len(hint))
    stderr_text = stderr_text + hint

    exit_code = proc.returncode or 0
    success = exit_code == 0
    data: dict[str, Any] = {
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "sandboxed": sandboxed,
        "truncated": stdout_truncated or stderr_truncated,
    }
    # Escape brackets so Rich does not swallow the sandbox tag
    display = f"exit={exit_code}" + (" \\[sandbox]" if sandboxed else "")
    if success:
        return ToolResult(True, data=data, display=display)
    # Non-zero exit: report failure so the model can't mistake a crashing
    # command for success. stdout/stderr remain in `data` so it can diagnose.
    err_tail = (stderr_text.strip() or stdout_text.strip())[-400:]
    return ToolResult(
        False,
        data=data,
        error=(
            f"command exited {exit_code}: {err_tail}" if err_tail else f"command exited {exit_code}"
        ),
        display=display,
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
)
 
SHELL_TOOLS: list[Tool] = [run_command]