"""Terminal-bench harness.
 
A Terminal-bench task is a CLI task with a description and a verification
step. This harness supports a simple task schema:
 
    {
      "id": "task-001",
      "description": "Create a Python script fib.py that prints fib(10).",
      "setup": ["pip install -q pytest"],                          # optional
      "verify": "python fib.py | grep -q '^55$'",                  # shell; exit 0 = pass
      "files": {"initial.txt": "..."},                             # optional seed files
      "timeout": 300
    }
 
Each task runs in a fresh temp working dir (so tasks are independent). Real
Terminal-bench tasks use a tmux container contract; you can adapt this harness
to spawn that contract by changing how ``setup`` / ``verify`` execute.
"""
 
from __future__ import annotations
 
import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
 
from squishy.api import Squishy
from squishy.bench.runner import BenchResult
 
log = logging.getLogger("squishy.bench.terminalbench")
 
 
@dataclass
class TerminalTask:
    id: str
    description: str
    verify: str = ""
    setup: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    timeout: float = 300.0
 
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TerminalTask:
        return cls(
            id=d["id"],
            description=d["description"],
            verify=d.get("verify", ""),
            setup=list(d.get("setup", []) or []),
            files=dict(d.get("files", {}) or {}),
            timeout=float(d.get("timeout", 300.0)),
        )
 
 
async def run_terminal_task(
    task: TerminalTask,
    *,
    squishy: Squishy,
    workspace_root: str | Path | None = None,
) -> BenchResult:
    """Run one Terminal-bench task. Returns pass/fail based on ``verify`` exit code."""
    if workspace_root is None:
        workspace = Path(tempfile.mkdtemp(prefix=f"squishy-term-{task.id}-"))
    else:
        workspace = Path(workspace_root) / task.id
        workspace.mkdir(parents=True, exist_ok=True)
 
    for relpath, content in task.files.items():
        full = workspace / relpath
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
 
    for cmd in task.setup:
        rc, out, err = await _run_shell(cmd, cwd=workspace)
        if rc != 0:
            return BenchResult(
                task_id=task.id,
                success=False,
                error=f"setup '{cmd}' exited {rc}: {err[:200]}",
            )
 
    try:
        task_result = await squishy.run(
            task.description, working_dir=str(workspace), timeout=task.timeout
        )
    except Exception as e:  # noqa: BLE001
        return BenchResult(task_id=task.id, success=False, error=f"agent: {type(e).__name__}: {e}")
 
    prediction: dict[str, Any] = {
        "task_id": task.id,
        "agent_success": task_result.success,
        "final_text": task_result.final_text,
        "turns": task_result.turns_used,
        "tokens": task_result.tokens_used,
        "files_created": task_result.files_created,
        "files_edited": task_result.files_edited,
        "commands_run": task_result.commands_run,
    }
 
    verified = True
    verify_output = ""
    if task.verify:
        rc, out, err = await _run_shell(task.verify, cwd=workspace)
        verified = rc == 0
        verify_output = (out + err)[-800:]
        prediction["verify_exit_code"] = rc
        prediction["verify_output"] = verify_output
 
    return BenchResult(
        task_id=task.id,
        success=task_result.success and verified,
        prediction=prediction,
        error="" if verified else f"verify failed: {verify_output[:200]}",
        elapsed_s=task_result.elapsed_s,
    )
 
 
def load_tasks(path: str | Path) -> list[TerminalTask]:
    """Load tasks from a JSONL or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    tasks: list[dict[str, Any]] = []
    if p.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    else:
        data = json.loads(text)
        tasks = data if isinstance(data, list) else [data]
    return [TerminalTask.from_dict(t) for t in tasks]
 
 
async def _run_shell(cmd: str, *, cwd: Path, timeout: float = 120.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"shell timeout after {timeout}s"
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )