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
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from squishy.agent import TaskResult
from squishy.api import Squishy
from squishy.bench.runner import BenchResult
from squishy.errors import AgentCancelled, AgentTimeout, LLMError

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


@dataclass
class ShellResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


class TerminalBackend(Protocol):
    async def prepare_workspace(
        self,
        task: TerminalTask,
        *,
        workspace_root: str | Path | None = None,
    ) -> Path: ...

    async def run_setup(self, task: TerminalTask, workspace: Path) -> list[ShellResult]: ...

    async def verify(self, task: TerminalTask, workspace: Path) -> ShellResult | None: ...


class LocalTerminalBackend:
    async def prepare_workspace(
        self,
        task: TerminalTask,
        *,
        workspace_root: str | Path | None = None,
    ) -> Path:
        if workspace_root is None:
            workspace = Path(tempfile.mkdtemp(prefix=f"squishy-term-{task.id}-"))
        else:
            workspace = Path(workspace_root) / task.id
            workspace.mkdir(parents=True, exist_ok=True)

        for relpath, content in task.files.items():
            full = (workspace / relpath).resolve()
            if not full.is_relative_to(workspace.resolve()):
                raise ValueError(f"path traversal in task files: {relpath}")
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
        return workspace

    async def run_setup(self, task: TerminalTask, workspace: Path) -> list[ShellResult]:
        results: list[ShellResult] = []
        for cmd in task.setup:
            result = await _run_shell(cmd, cwd=workspace, timeout=task.timeout)
            results.append(result)
            if result.exit_code != 0:
                break
        return results

    async def verify(self, task: TerminalTask, workspace: Path) -> ShellResult | None:
        if not task.verify:
            return None
        return await _run_shell(task.verify, cwd=workspace, timeout=task.timeout)


async def run_terminal_task(
    task: TerminalTask,
    *,
    squishy: Squishy,
    workspace_root: str | Path | None = None,
    backend: TerminalBackend | None = None,
) -> BenchResult:
    """Run one Terminal-bench task. Returns pass/fail based on ``verify`` exit code."""
    runner = backend or LocalTerminalBackend()
    workspace = await runner.prepare_workspace(task, workspace_root=workspace_root)

    setup_results = await runner.run_setup(task, workspace)
    for result in setup_results:
        if result.exit_code != 0:
            return BenchResult(
                task_id=task.id,
                success=False,
                error=f"setup '{result.command}' exited {result.exit_code}: {result.stderr[:200]}",
                artifacts={
                    "workspace": str(workspace),
                    "setup_results": [item.to_dict() for item in setup_results],
                },
            )

    try:
        task_result = await squishy.run(
            task.description, working_dir=str(workspace), timeout=task.timeout
        )
    except AgentTimeout as e:
        return BenchResult(
            task_id=task.id,
            success=False,
            error=f"agent_timeout: {e}",
            prediction={"error_class": "agent_timeout"},
        )
    except AgentCancelled as e:
        return BenchResult(
            task_id=task.id,
            success=False,
            error=f"agent_cancelled: {e}",
            prediction={"error_class": "agent_cancelled"},
        )
    except LLMError as e:
        return BenchResult(
            task_id=task.id,
            success=False,
            error=f"llm_error: {e}",
            prediction={"error_class": "llm_error"},
        )
    except Exception as e:  # noqa: BLE001
        return BenchResult(
            task_id=task.id,
            success=False,
            error=f"agent: {type(e).__name__}: {e}",
            prediction={"error_class": "agent_exception"},
        )

    prediction: dict[str, Any] = {
        "task_id": task.id,
        "agent_success": task_result.success,
        "final_text": task_result.final_text,
        "turns": task_result.turns_used,
        "tokens": task_result.tokens_used,
        "files_created": task_result.files_created,
        "files_edited": task_result.files_edited,
        "commands_run": task_result.commands_run,
        "plan_state": task_result.plan_state,
        "plan_adherence": _plan_adherence(task_result.plan_state),
        "tool_counts": _tool_counts(task_result.messages),
    }
    artifacts: dict[str, Any] = {
        "workspace": str(workspace),
        "transcript": task_result.messages,
        "setup_results": [item.to_dict() for item in setup_results],
        "workspace_snapshot": _snapshot_workspace(workspace),
    }

    verified = True
    verify_result = await runner.verify(task, workspace)
    verify_output = ""
    if verify_result is not None:
        verified = verify_result.exit_code == 0
        verify_output = (verify_result.stdout + verify_result.stderr)[-800:]
        prediction["verify_exit_code"] = verify_result.exit_code
        prediction["verify_output"] = verify_output
        artifacts["verify_result"] = verify_result.to_dict()

    prediction["error_class"] = _classify_error(task_result, verified)

    if not verified:
        error_msg = f"verify failed: {verify_output[:200]}"
    elif not task_result.success:
        error_msg = task_result.error or ""
    else:
        error_msg = ""

    return BenchResult(
        task_id=task.id,
        success=task_result.success and verified,
        prediction=prediction,
        error=error_msg,
        elapsed_s=task_result.elapsed_s,
        artifacts=artifacts,
    )


def _plan_adherence(plan_state: dict[str, Any] | None) -> dict[str, Any]:
    """Compute a compact summary of how closely the agent followed its plan."""
    if not plan_state:
        return {"had_plan": False}
    progress = plan_state.get("progress") or {}
    total = int(progress.get("total") or 0)
    done = int(progress.get("done") or 0)
    blocked = int(progress.get("blocked") or 0)
    skipped = int(progress.get("skipped") or 0)
    pending = int(progress.get("pending") or 0)
    completion_ratio = (done / total) if total else 0.0
    return {
        "had_plan": True,
        "approved": bool(plan_state.get("approved")),
        "total_steps": total,
        "done": done,
        "blocked": blocked,
        "skipped": skipped,
        "pending": pending,
        "completion_ratio": round(completion_ratio, 3),
    }


def _tool_counts(messages: list[dict[str, Any]] | None) -> dict[str, int]:
    """Count tool calls per tool name from the recorded transcript."""
    counts: Counter[str] = Counter()
    for msg in messages or []:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name") or tc.get("name")
            if isinstance(name, str):
                counts[name] += 1
    return dict(counts)


def _classify_error(task_result: TaskResult, verified: bool) -> str:
    """Map the task outcome to a single-token failure class for easy triage."""
    if task_result.success and verified:
        return "ok"
    if not verified:
        return "verify_failed"
    err = (task_result.error or "").lower()
    if "plan_task" in err or "plan-mode" in err:
        return "plan_mode_no_plan"
    if "max turns" in err:
        return "max_turns"
    if "consecutive tool failures" in err:
        return "tool_error_loop"
    if "timeout" in err:
        return "agent_timeout"
    if err:
        return "llm_error"
    return "unknown"
 
 
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
 
 
def _snapshot_workspace(workspace: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        result.append({"path": str(path.relative_to(workspace)), "size": size})
    return result


async def _run_shell(cmd: str, *, cwd: Path, timeout: float = 120.0) -> ShellResult:
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
        return ShellResult(
            command=cmd,
            exit_code=124,
            stdout="",
            stderr=f"shell timeout after {timeout}s",
            timed_out=True,
        )
    return ShellResult(
        command=cmd,
        exit_code=proc.returncode if proc.returncode is not None else 0,
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
    )
