"""SWE-bench harness.
 
Workflow per instance:
  1. Prepare a clean workspace: git clone the instance's repo, checkout base_commit
  2. Build a prompt from problem_statement (+ hints_text if present)
  3. Run the agent with working_dir=workspace, timeout=task_timeout
  4. `git diff` base_commit..HEAD in the workspace -> ``model_patch``
  5. Return a prediction dict compatible with SWE-bench evaluation:
     {"instance_id", "model_name_or_path", "model_patch"}
 
Evaluation itself (running the patched repo against the hidden tests in Docker)
is delegated to the upstream harness:
    python -m swebench.harness.run_evaluation \\
        --predictions_path predictions.jsonl \\
        --dataset_name princeton-nlp/SWE-bench_Lite
"""
 
from __future__ import annotations
 
import asyncio
import logging
import os
from pathlib import Path
from typing import Any
 
from squishy.api import Squishy
from squishy.bench.runner import BenchResult
from squishy.errors import BenchError
 
log = logging.getLogger("squishy.bench.swebench")
 
# SWE-bench instances use github.com/<repo> with a specific base_commit.
GITHUB_TEMPLATE = "https://github.com/{repo}.git"
 
 
async def prepare_workspace(instance: dict[str, Any], root: str | Path) -> Path:
    """Clone the instance repo at base_commit into ``root/<instance_id>``.
 
    Re-uses an existing clone if the commit already matches; otherwise
    fetches and resets. Idempotent enough to retry cheaply.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    dest = Path(root) / instance_id
 
    if (dest / ".git").exists():
        rc, _, _ = await _git(["rev-parse", "HEAD"], cwd=dest)
        if rc == 0:
            await _git(["fetch", "--depth=1", "origin", base_commit], cwd=dest)
            rc, _, err = await _git(["reset", "--hard", base_commit], cwd=dest)
            if rc != 0:
                raise BenchError(f"reset to {base_commit} failed: {err}")
            await _git(["clean", "-fdx"], cwd=dest)
            return dest
 
    dest.mkdir(parents=True, exist_ok=True)
    url = GITHUB_TEMPLATE.format(repo=repo)
    rc, _, err = await _git(["clone", url, str(dest)])
    if rc != 0:
        raise BenchError(f"clone {url} failed: {err}")
    rc, _, err = await _git(["checkout", base_commit], cwd=dest)
    if rc != 0:
        raise BenchError(f"checkout {base_commit} failed: {err}")
    return dest
 
 
def build_prompt(instance: dict[str, Any]) -> str:
    """Compose a prompt from the SWE-bench instance fields."""
    parts = [
        "You are fixing a bug in a real Python repository. The working directory is a clone",
        "of the repository at the relevant commit. Read the relevant files, make the smallest",
        "change that fixes the problem, and verify by running tests.",
        "",
        "## Problem",
        instance.get("problem_statement", "").strip(),
    ]
    hints = (instance.get("hints_text") or "").strip()
    if hints:
        parts += ["", "## Hints", hints]
    return "\n".join(parts)
 
 
async def capture_patch(workspace: Path, base_commit: str) -> str:
    """Return a unified diff of workspace changes since ``base_commit``."""
    await _git(["add", "-A"], cwd=workspace)
    rc, out, err = await _git(
        ["diff", "--no-color", base_commit, "--"], cwd=workspace
    )
    if rc != 0:
        raise BenchError(f"git diff failed: {err}")
    return out
 
 
async def run_swebench_instance(
    instance: dict[str, Any],
    *,
    squishy: Squishy,
    workspace_root: str | Path,
    model_name: str,
    task_timeout: float = 900.0,
) -> BenchResult:
    """Run one SWE-bench instance end-to-end, returning a prediction record."""
    instance_id = instance["instance_id"]
    try:
        workspace = await prepare_workspace(instance, workspace_root)
    except BenchError as e:
        return BenchResult(task_id=instance_id, success=False, error=f"workspace: {e}")
 
    prompt = build_prompt(instance)
    try:
        task_result = await squishy.run(
            prompt, working_dir=str(workspace), timeout=task_timeout
        )
    except Exception as e:  # noqa: BLE001
        return BenchResult(task_id=instance_id, success=False, error=f"agent: {type(e).__name__}: {e}")
 
    try:
        patch = await capture_patch(workspace, instance["base_commit"])
    except BenchError as e:
        return BenchResult(task_id=instance_id, success=False, error=f"diff: {e}")
 
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }
    return BenchResult(
        task_id=instance_id,
        success=task_result.success and bool(patch.strip()),
        prediction=prediction,
        error="" if task_result.success else task_result.error,
        elapsed_s=task_result.elapsed_s,
    )
 
 
async def _git(args: list[str], *, cwd: str | Path | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd) if cwd else None,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )