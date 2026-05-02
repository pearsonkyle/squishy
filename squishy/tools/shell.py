"""Async shell execution tool. Docker sandbox when available, subprocess otherwise."""
 
from __future__ import annotations
 
import asyncio
import os
import re
import shutil
from typing import Any
 
from squishy.tools.base import Tool, ToolContext, ToolResult
 
DEFAULT_TIMEOUT = 60
OUTPUT_CAP_STDOUT = 8000
OUTPUT_CAP_STDERR = 4000
# Max lines to keep per individual failure block in pytest output.
_PYTEST_FAILURE_BLOCK_CAP = 40
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
 
 
def _looks_like_pytest(text: str) -> bool:
    """Heuristic: does this look like pytest output?"""
    markers = ("= FAILURES =", "short test summary info", "FAILED ", "passed", "failed")
    count = sum(1 for m in markers if m in text)
    return count >= 2


def _smart_cap_pytest(raw: bytes, cap: int) -> tuple[str, bool]:
    """Truncate pytest output while preserving failure details.

    Keeps: failure blocks (FAILURES section), short test summary, final
    summary line. Trims: passing test dots/lines and verbose pass output.
    Falls back to generic _cap_output if the result is still too large.
    """
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= cap:
        return text, False

    lines = text.splitlines()
    kept: list[str] = []
    in_failures_section = False
    in_failure_block = False
    failure_block_lines = 0
    in_summary = False

    for line in lines:
        # Detect the FAILURES section header
        if "= FAILURES =" in line or "= ERRORS =" in line:
            in_failures_section = True
            in_failure_block = False
            kept.append(line)
            continue

        # Detect individual failure block headers (underlined test names)
        if in_failures_section and line.startswith("_") and line.endswith("_"):
            in_failure_block = True
            failure_block_lines = 0
            kept.append(line)
            continue

        # End of FAILURES section (next separator line)
        if in_failures_section and line.startswith("=") and ("passed" in line or "failed" in line or "error" in line):
            in_failures_section = False
            in_failure_block = False
            kept.append(line)
            continue

        # Keep failure block lines (capped per block)
        if in_failure_block:
            failure_block_lines += 1
            if failure_block_lines <= _PYTEST_FAILURE_BLOCK_CAP:
                kept.append(line)
            elif failure_block_lines == _PYTEST_FAILURE_BLOCK_CAP + 1:
                kept.append("    ... (failure block truncated)")
            continue

        # Keep non-block lines in failures section (e.g., between blocks)
        if in_failures_section:
            kept.append(line)
            continue

        # Detect short test summary section
        if "short test summary info" in line:
            in_summary = True
            kept.append(line)
            continue

        if in_summary:
            kept.append(line)
            continue

        # Keep lines with FAILED, ERROR, or warnings
        if "FAILED " in line or "ERROR " in line or "ERRORS " in line:
            kept.append(line)
            continue

        # Keep the final summary line (e.g., "5 failed, 120 passed in 45s")
        if line.startswith("=") and ("passed" in line or "failed" in line or "error" in line):
            kept.append(line)
            continue

        # Keep collection/setup errors
        if "CollectionError" in line or "ImportError" in line:
            kept.append(line)
            continue

    result = "\n".join(kept)
    if len(result) > cap:
        # Still too large — fall back to generic tail truncation.
        # Use the already-decoded text to avoid re-decoding raw.
        dropped = len(text) - cap
        marker = f"…<truncated {dropped} bytes of head>\n"
        return marker + text[-cap:], True

    # Prepend a marker if we dropped content
    dropped = len(text) - len(result)
    if dropped > 0:
        marker = f"…<{dropped} chars of passing output trimmed>\n"
        result = marker + result

    return result, dropped > 0


# Regex for "short test summary info" FAILED lines.
# Format: FAILED path/test.py::TestClass::test_name - ErrorType: message
_SUMMARY_FAILED_RE = re.compile(
    r"^FAILED\s+(\S+)\s*-\s*(\w+(?:Error|Exception|Warning|Failure)?:?\s*.*)$",
    re.MULTILINE,
)
# Regex for the final pytest summary line counts.
_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error)")


def _extract_test_failures(output: str) -> dict[str, Any] | None:
    """Parse pytest output into a structured test summary.

    Returns a dict with keys ``failures``, ``passed``, ``failed``, ``errors``
    or *None* if the output doesn't look like pytest.
    """
    if not _looks_like_pytest(output):
        return None

    failures: list[dict[str, str]] = []

    # Strategy 1: Parse "short test summary info" FAILED lines (most reliable).
    for m in _SUMMARY_FAILED_RE.finditer(output):
        test_name = m.group(1)
        error_msg = m.group(2).strip()
        # Cap error message to keep nudge messages concise.
        if len(error_msg) > 120:
            error_msg = error_msg[:117] + "..."
        failures.append({"test": test_name, "error": error_msg})

    # Strategy 2 fallback: Parse FAILURES section headers if no summary lines.
    if not failures:
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("_") and stripped.endswith("_") and len(stripped) > 4:
                # Extract test name from "_____ test_name _____"
                test_name = stripped.strip("_ ").strip()
                if test_name:
                    failures.append({"test": test_name, "error": ""})

    # Parse counts from the final summary line.
    passed = 0
    failed = 0
    errors = 0
    for m in _SUMMARY_COUNT_RE.finditer(output):
        count = int(m.group(1))
        kind = m.group(2)
        if kind == "passed":
            passed = count
        elif kind == "failed":
            failed = count
        elif kind == "error":
            errors = count

    # If we found no counts and no failures, not useful.
    if passed == 0 and failed == 0 and errors == 0 and not failures:
        return None

    return {
        "failures": failures[:10],  # Cap at 10 to keep messages manageable
        "passed": passed,
        "failed": failed,
        "errors": errors,
    }


def _docker_available() -> bool:
    return shutil.which("docker") is not None
 
 
async def _run_command(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str):
        return ToolResult(False, error="`command` is required (string)")
    timeout = float(args.get("timeout") or DEFAULT_TIMEOUT)
    raw_cwd = args.get("cwd")
    if raw_cwd and isinstance(raw_cwd, str):
        from squishy.tools.fs import _safe_resolve
        cwd, cwd_err = _safe_resolve(raw_cwd, ctx.working_dir)
        if cwd_err:
            return ToolResult(False, error=cwd_err)
    else:
        cwd = ctx.working_dir
 
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
        # Merge per-instance extra_env (e.g. PYTHONUSERBASE) if set,
        # otherwise inherit parent environment naturally.
        if ctx.extra_env:
            exec_env = {**os.environ, **ctx.extra_env}
        else:
            exec_env = None

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

    # Use smart pytest truncation when output looks like pytest.
    # Bench mode uses a tighter cap to reduce token waste.
    stdout_cap = 6000 if ctx.permission_mode == "bench" else OUTPUT_CAP_STDOUT
    stdout_decoded = stdout_b.decode("utf-8", errors="replace")
    test_summary = None
    if _looks_like_pytest(stdout_decoded):
        stdout_text, stdout_truncated = _smart_cap_pytest(stdout_b, stdout_cap)
        test_summary = _extract_test_failures(stdout_decoded)
    else:
        stdout_text, stdout_truncated = _cap_output(stdout_b, stdout_cap)
    stderr_text, stderr_truncated = _cap_output(stderr_b, OUTPUT_CAP_STDERR - len(hint))
    stderr_text = stderr_text + hint

    exit_code = proc.returncode  # always set after communicate()
    success = exit_code == 0
    data: dict[str, Any] = {
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "sandboxed": sandboxed,
        "truncated": stdout_truncated or stderr_truncated,
    }
    if test_summary:
        data["test_summary"] = test_summary
    # Escape brackets so Rich does not swallow the sandbox tag
    _esc = "\\"
    display = f"exit={exit_code}" + (f" {_esc}[sandbox]" if sandboxed else "")
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