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
import json as _json
import logging
import os
import re
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
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", base_commit):
        raise BenchError(f"invalid base_commit (must be hex SHA): {base_commit!r}")
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
        # Commit may be old; try fetching it explicitly.
        await _git(["fetch", "origin", base_commit], cwd=dest)
        rc, _, err = await _git(["checkout", base_commit], cwd=dest)
        if rc != 0:
            raise BenchError(f"checkout {base_commit} failed: {err}")
    return dest
 
 
def build_prompt(instance: dict[str, Any]) -> str:
    """Compose a prompt from the SWE-bench instance fields."""
    parts = [
        "You are fixing a bug in a real Python repository. The working directory is a",
        "clone of the repository at the relevant commit.",
        "",
        "## CRITICAL: Action Bias",
        "You MUST attempt an `edit_file` fix within your first 5 tool calls. Do NOT",
        "spend excessive time exploring. Read the problem, locate the relevant file,",
        "and make your best fix attempt. You can always iterate if the fix is wrong.",
        "A wrong fix that you iterate on is MUCH better than perfect understanding",
        "with no fix attempt.",
        "",
        "## Workflow",
        "1. **Understand** (1 call): Read the problem statement. Extract the file path",
        "   and function/class name from the traceback or description.",
        "2. **Locate** (1-2 calls): Find the file and function to fix.",
        "   - BEST: call `recall(query='class_name')` or `recall(query='function_name')` to",
        "     instantly locate symbols in the codebase. This is much faster than searching.",
        "   - If the problem says `foo.bar.baz`, read `foo/bar.py` directly.",
        "   - If a traceback is included, the BOTTOM frame shows the file and line.",
        "   - Use `read_file(path=..., offset=<line-50>, limit=100)` to see context.",
        "   - Only use `search_files` if `recall` returns no results and you cannot",
        "     determine the file from the problem statement.",
        "3. **Fix** (1-2 calls): Make the smallest change that fixes the bug:",
        "   - Use `edit_file(path=..., old_str=..., new_str=...)` — NEVER `write_file`.",
        "   - Include 2-3 lines of surrounding context in `old_str` for uniqueness.",
        "   - If `edit_file` fails, call `read_file` on those exact lines and retry.",
        "4. **Verify** (1 call): Run the specific failing test:",
        "   - `run_command(command=\"python -m pytest path/to/test.py::test_name -xvs\")`",
        "   - If it passes, you're done — go to step 5 IMMEDIATELY.",
        "   - If it fails, read the error and fix again.",
        "5. **Finish**: When your fix works (test passes OR verification confirms the",
        "   fix), respond with ONLY a plain text summary of what you changed and why.",
        "   Do NOT call any more tools. Do NOT run more verification commands.",
        "   Just write text and stop.",
        "",
        "## Rules",
        "- **DO NOT** spend more than 3 calls on exploration before making an edit.",
        "- Make the MINIMAL change needed. Do not refactor or clean up.",
        "- Read files before editing. Never guess file content.",
        "- Do not install packages or modify setup.py/pyproject.toml.",
        "- After locating the bug, call `save_note(key=\"bug_location\",",
        "  content=\"file:line — description\")` so you remember it.",
        "- If `edit_file` fails with a hint showing actual content, use that",
        "  EXACT text for your next call. Do not reconstruct from memory.",
        "- If a test fails after your fix, read the test to understand expectations.",
        "- Use `show_diff` before finishing to verify all changes look correct.",
        "- If the problem includes a reproduction script, you may run it to confirm.",
        "- NEVER run `git log`, `git show`, `git blame` — they waste turns.",
        "  The bug is described in the problem statement; fix it from the source code.",
        "- NEVER run the full test suite. Only run the specific test.",
        "- If stuck after 3 attempts, re-read the problem and try a different file or approach.",
        "- If the problem doesn't mention a specific file, use `search_files` with a",
        "  keyword from the error message or the class/function name to locate the code.",
        "",
        "## Environmental Error Handling",
        "- If running tests produces ImportError or ModuleNotFoundError unrelated to",
        "  your fix (e.g., `collections.Mapping` removed in Python 3.10+, missing optional",
        "  dependencies, or `typing` changes), DO NOT fix those import errors — they are",
        "  NOT the bug you need to fix.",
        "- Instead, run a more targeted test: `python -m pytest path/to/test.py::TestClass::test_method -x`",
        "- Or write a small reproduction script that directly tests the function you fixed.",
        "- NEVER edit files just to fix import compatibility issues.",
        "- If the entire test module fails to import, find a different test or write a",
        "  minimal script to verify your fix.",
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
        ["diff", "--cached", "--no-color", base_commit, "--"], cwd=workspace
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
    auto_init: bool = False,
) -> BenchResult:
    """Run one SWE-bench instance end-to-end, returning a prediction record."""
    instance_id = instance["instance_id"]
    try:
        workspace = await prepare_workspace(instance, workspace_root)
    except BenchError as e:
        return BenchResult(task_id=instance_id, success=False, error=f"workspace: {e}")

    # Build structural index so the agent can use `recall` to locate code.
    if auto_init:
        try:
            from squishy.index import _build_index_async, save_agents_md
            from squishy.index.store import save_index

            import time as _time
            t0 = _time.monotonic()
            idx = await _build_index_async(str(workspace), prior=None, concurrency=8)
            save_index(str(workspace), idx)
            save_agents_md(idx, str(workspace))
            dt = _time.monotonic() - t0
            stats = idx.meta.stats
            log.info(
                "indexed %s: %d files, %d symbols in %.1fs",
                instance_id, stats.get("files", 0), stats.get("symbols", 0), dt,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("index build failed for %s: %s", instance_id, e)

    # Create a session for this bench instance so conversations can be
    # inspected later and exported as training data.
    session_id: str | None = None
    try:
        from squishy.session import create_session
        sess = create_session(
            model=model_name,
            working_dir=str(workspace),
            mode="bench",
        )
        session_id = sess.id
    except Exception:  # noqa: BLE001
        log.debug("session creation failed for %s", instance_id, exc_info=True)

    prompt = build_prompt(instance)
    try:
        task_result = await squishy.run(
            prompt, working_dir=str(workspace), timeout=task_timeout,
            session_id=session_id,
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
    # A non-empty patch counts as success even if the agent hit max turns,
    # because the fix was applied — the agent just didn't stop cleanly.
    has_patch = bool(patch.strip())

    # Collect diagnostics from the task result for post-hoc analysis.
    diagnostics = _extract_diagnostics(task_result)
    if session_id:
        diagnostics["session_id"] = session_id
    # Include turn_log for all instances (compact structured data).
    turn_log = getattr(task_result, "turn_log", [])
    if turn_log:
        diagnostics["turn_log"] = turn_log
    # Include transcript for failed instances so we can inspect agent behavior.
    if not has_patch:
        diagnostics["transcript"] = getattr(task_result, "messages", []) or []

    return BenchResult(
        task_id=instance_id,
        success=has_patch,
        prediction=prediction,
        artifacts=diagnostics,
        error="" if has_patch else (task_result.error or "empty patch"),
        elapsed_s=task_result.elapsed_s,
    )
 
 
def _extract_diagnostics(task_result: Any) -> dict[str, Any]:
    """Extract tool-usage diagnostics from a TaskResult for post-hoc analysis.

    Most counters come directly from TaskResult (accurate, from _LoopState).
    Message-walking is only used for read_paths/re_reads/cache_hits/system_nudges
    which require inspecting individual messages.
    """
    messages = getattr(task_result, "messages", []) or []
    read_paths: dict[str, int] = {}
    cache_hits = 0
    system_nudges = 0

    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                if func.get("name") == "read_file":
                    try:
                        args = _json.loads(func.get("arguments", "{}"))
                        path = args.get("path", "?")
                        read_paths[path] = read_paths.get(path, 0) + 1
                    except Exception:  # noqa: BLE001
                        pass
        elif msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("[system]"):
                system_nudges += 1
        elif msg.get("role") == "tool" and msg.get("name") == "read_file":
            content = msg.get("content", "")
            if "cache_hit" in content:
                cache_hits += 1

    return {
        "turns_used": getattr(task_result, "turns_used", 0),
        "tokens_used": getattr(task_result, "tokens_used", 0),
        "tool_counts": getattr(task_result, "tool_call_counts", {}),
        "files_read": len(read_paths),
        "re_reads": {k: v for k, v in read_paths.items() if v > 1},
        "cache_hits": cache_hits,
        "files_edited": getattr(task_result, "files_edited", []),
        "commands_run": getattr(task_result, "commands_run", 0),
        "empty_responses": getattr(task_result, "empty_responses", 0),
        "prose_responses": getattr(task_result, "prose_completions", 0),
        "system_nudges": system_nudges,
        "quality_skips": getattr(task_result, "quality_skips", 0),
        "final_phase": getattr(task_result, "final_phase", ""),
        "explore_turns": getattr(task_result, "explore_turns", 0),
        "fix_verify_cycles": getattr(task_result, "fix_verify_cycles", 0),
        "quality_violations": getattr(task_result, "total_quality_violations", 0),
        "edit_failures": getattr(task_result, "edit_failures", 0),
    }


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
        proc.returncode if proc.returncode is not None else 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )