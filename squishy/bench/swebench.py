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
            await _git(["clean", "-fdx", "-e", ".squishy"], cwd=dest)
            _ensure_squishy_gitignore(dest)
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

    # Ensure .squishy/ (auto-init index) is git-ignored so it doesn't
    # pollute the patch captured after the agent run.
    _ensure_squishy_gitignore(dest)

    return dest


def _snapshot_pth_files(site_dir: str) -> set[str]:
    """Return the set of .pth filenames currently in *site_dir*."""
    try:
        return {f for f in os.listdir(site_dir) if f.endswith(".pth")}
    except OSError:
        return set()


def _cleanup_leaked_pth(site_dir: str, before: set[str]) -> None:
    """Remove .pth files that appeared in *site_dir* since *before* snapshot.

    ``pip install --user`` is supposed to write only to PYTHONUSERBASE, but
    namespace-package ``.pth`` files (e.g. ``distutils-precedence.pth``,
    ``geocat_comp-0.2-nspkg.pth``) leak into the *system* site-packages
    directory.  These corrupt Python for every subsequent process in the
    container.  Removing them immediately after each install prevents
    cross-instance contamination.
    """
    after = _snapshot_pth_files(site_dir)
    leaked = after - before
    for name in leaked:
        target = os.path.join(site_dir, name)
        try:
            os.remove(target)
            log.debug("removed leaked .pth: %s", target)
        except OSError:
            pass


async def install_deps(instance: dict[str, Any], workspace: Path) -> str | None:
    """Run install_config.install commands in an isolated prefix.

    V2 instances carry ``install_config.install`` — a list of shell commands
    (usually pip install) that must run before tests work.  Running these
    *before* the agent starts prevents wasted turns on ImportError /
    ModuleNotFoundError failures.

    Each instance gets its own ``PYTHONUSERBASE`` directory so packages from
    one instance cannot leak into another.  The caller must set
    ``os.environ["PYTHONUSERBASE"]`` to the returned path (and add its
    ``bin/`` to ``PATH``) before launching the agent, then clean up after.

    Returns the PYTHONUSERBASE path used, or None if no installs were needed.
    """
    install_config = instance.get("install_config") or {}
    cmds: list[str] = install_config.get("install") or []
    if not cmds:
        return None

    if isinstance(cmds, str):
        cmds = [cmds]

    # Isolated prefix for this instance — prevents dep leakage.
    user_base = str(workspace / ".user_packages")
    os.makedirs(user_base, exist_ok=True)

    env = os.environ.copy()
    env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    env["PYTHONUSERBASE"] = user_base
    # Ensure pip-installed scripts (pytest plugins, etc.) are on PATH.
    env["PATH"] = f"{user_base}/bin:{env.get('PATH', '')}"

    # Snapshot system site-packages .pth files before each install so we can
    # detect and remove any that leak during pip install --user.
    import site as _site
    sys_site_dirs = _site.getsitepackages()

    for cmd in cmds:
        if cmd.strip().startswith("pip install") and "--user" not in cmd:
            # Editable installs (-e) don't work with --user, so drop -e.
            if " -e " in cmd or " -e." in cmd:
                cmd = cmd.replace(" -e ", " ").replace(" -e.", " .")
            cmd = cmd.replace("pip install", "pip install --user", 1)

        # Snapshot .pth files before this command.
        pth_snapshots = {d: _snapshot_pth_files(d) for d in sys_site_dirs}

        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", cmd,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            log.warning("install timeout for %s: %s", instance["instance_id"], cmd[:80])
            # Clean up leaked .pth files even on timeout.
            for d, snap in pth_snapshots.items():
                _cleanup_leaked_pth(d, snap)
            return user_base

        # Clean up any .pth files that leaked into system site-packages.
        for d, snap in pth_snapshots.items():
            _cleanup_leaked_pth(d, snap)

        if proc.returncode != 0:
            err = stderr_b.decode("utf-8", errors="replace")[:200]
            log.warning(
                "install failed for %s (rc=%d): %s → %s",
                instance["instance_id"], proc.returncode, cmd[:80], err,
            )
            # Continue — some commands may fail non-fatally (e.g. optional extras)
    return user_base
 
 
_GITIGNORE_ENTRIES = (
    ".squishy/",
    ".user_packages/",
    "*.egg-info/",
    "build/",
    "__pycache__/",
    ".eval_venv/",
)


def _ensure_squishy_gitignore(workspace: Path) -> None:
    """Add agent/build artifacts to .gitignore so they don't pollute patches."""
    gitignore = workspace / ".gitignore"
    if gitignore.exists():
        text = gitignore.read_text(encoding="utf-8", errors="replace")
    else:
        text = ""

    added = False
    for entry in _GITIGNORE_ENTRIES:
        if entry not in text:
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"{entry}\n"
            added = True

    if added:
        gitignore.write_text(text, encoding="utf-8")


def _recall_from_index(workspace: str | Path, problem_text: str, limit: int = 8) -> list[dict]:
    """Pre-query the repo index using the problem statement.

    Returns a list of {kind, name, path, lines, summary} dicts for the
    most relevant symbols/files, giving the agent a head start on where
    to look and edit.
    """
    try:
        from squishy.index.store import load_index
        from squishy.tools.recall import _score, _tokens, _trim
    except ImportError:
        return []

    idx = load_index(str(workspace))
    if idx is None:
        return []

    q_lower = problem_text.strip().lower()[:500]
    q_tokens = _tokens(problem_text)
    if not q_tokens:
        return []

    scored: list[tuple[float, Any]] = []
    for node in idx.root.walk():
        if node.kind == "repo":
            continue
        s = _score(node, q_lower, q_tokens)
        if s > 0:
            scored.append((s, node))
    scored.sort(key=lambda t: (-t[0], t[1].path, t[1].name))

    # Prefer symbols (class/function/method) over files/dirs
    results: list[dict] = []
    seen_paths: set[str] = set()
    for _, node in scored[:limit * 3]:
        if len(results) >= limit:
            break
        key = f"{node.path}:{node.name}"
        if key in seen_paths:
            continue
        seen_paths.add(key)
        d = _trim(node, depth=0)
        results.append(d)
    return results


def build_prompt(instance: dict[str, Any], *, workspace: str | Path | None = None) -> str:
    """Compose a user prompt from the SWE-bench instance fields.

    Workflow instructions live in the system prompt (via ``_mode_block("bench")``
    in context.py). This user message contains ONLY the problem statement,
    failing tests, hints, and (optionally) index-based code pointers — keeping
    it small so it survives trimming.
    """
    parts = [
        "Fix the bug described below. The working directory is a clone of the",
        "repository at the relevant commit.",
        "",
        "## Problem",
        instance.get("problem_statement", "").strip(),
    ]

    # Include the exact failing tests so the agent knows what to run.
    fail_to_pass = instance.get("FAIL_TO_PASS") or []
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = _json.loads(fail_to_pass)
        except (ValueError, TypeError):
            fail_to_pass = [fail_to_pass] if fail_to_pass.strip() else []
    if fail_to_pass:
        parts += [
            "",
            "## Failing Tests",
            "These are the specific tests that must PASS after your fix:",
        ]
        for test in fail_to_pass[:5]:  # cap at 5 to avoid prompt bloat
            parts.append(f"- `{test}`")
        if len(fail_to_pass) > 5:
            parts.append(f"- _(and {len(fail_to_pass) - 5} more)_")
        # Suggest a run command — prefer V2's install_config.test_cmd if available.
        install_config = instance.get("install_config") or {}
        test_cmd = install_config.get("test_cmd", "")
        if test_cmd:
            # Escape internal quotes so the prompt renders cleanly.
            escaped_cmd = test_cmd.replace('"', '\\"')
            parts.append(
                f"\nTo verify your fix, run: "
                f'`run_command(command="{escaped_cmd}")`'
            )
        else:
            test_args = " ".join(fail_to_pass[:3])
            parts.append(
                f"\nTo verify your fix, run: "
                f"`run_command(command=\"python -m pytest {test_args} -xvs\")`"
            )

    # Extract traceback file hints from the problem statement.
    problem_text = instance.get("problem_statement", "")
    tb_files = _extract_traceback_files(problem_text)
    if tb_files:
        parts += ["", "## Traceback File Hints"]
        parts.append("The traceback in the problem points to these files:")
        for fpath, line_no in tb_files[:3]:
            if line_no:
                parts.append(f"- `{fpath}` line {line_no}")
            else:
                parts.append(f"- `{fpath}`")
        parts.append("Start by reading the bottom-most file/line from the traceback.")

    hints = (instance.get("hints_text") or "").strip()
    if hints:
        parts += ["", "## Hints", hints]

    # Pre-query the index to give the agent code pointers.
    if workspace:
        problem_text = instance.get("problem_statement", "")
        pointers = _recall_from_index(workspace, problem_text, limit=8)
        if pointers:
            parts += ["", "## Relevant Code (from index)"]
            for p in pointers:
                line_info = ""
                if p.get("lines"):
                    line_info = f" (L{p['lines'][0]}-{p['lines'][1]})"
                summary = f" — {p['summary']}" if p.get("summary") else ""
                parts.append(f"- `{p['path']}`{line_info}: {p.get('kind', 'file')} `{p['name']}`{summary}")
            parts.append(
                "\nUse `read_file` on the most relevant file above to understand "
                "the code, then call `edit_file` with your fix."
            )

    return "\n".join(parts)
 
 
_PATCH_EXCLUDE_PATTERNS = (
    ":(exclude).squishy",
    ":(exclude).gitignore",
    ":(exclude).user_packages",
    ":(exclude)*.egg-info",
    ":(exclude)build",
    ":(exclude)__pycache__",
    ":(exclude).eval_venv",
    ":(exclude)venv",
    ":(exclude).venv",
    ":(exclude)dist",
    ":(exclude)*.pyc",
    ":(exclude)*.pyo",
    ":(exclude)*.so",
    ":(exclude).eggs",
    ":(exclude).tox",
)

# Maximum patch size in bytes.  Patches larger than this are almost
# certainly polluted with build artifacts / pip-installed packages.
_MAX_PATCH_BYTES = 512_000  # 500 KB


async def capture_patch(
    workspace: Path, base_commit: str,
    *, edited_files: list[str] | None = None,
) -> str:
    """Return a unified diff of workspace changes since ``base_commit``.

    Excludes common build artifacts and agent-generated directories.
    If the raw diff is too large (>500 KB), falls back to diffing only
    the files the agent explicitly edited via ``edit_file``.
    """
    await _git(["add", "-A"], cwd=workspace)
    rc, out, err = await _git(
        [
            "diff", "--cached", "--no-color", base_commit,
            "--", ".", *_PATCH_EXCLUDE_PATTERNS,
        ],
        cwd=workspace,
    )
    if rc != 0:
        raise BenchError(f"git diff failed: {err}")

    # If the patch is excessively large, restrict to agent-edited files only.
    if len(out.encode()) > _MAX_PATCH_BYTES and edited_files:
        log.warning(
            "patch too large (%d bytes), restricting to %d edited files",
            len(out.encode()), len(edited_files),
        )
        rc2, out2, err2 = await _git(
            ["diff", "--cached", "--no-color", base_commit, "--"]
            + list(edited_files),
            cwd=workspace,
        )
        if rc2 == 0 and out2.strip():
            return out2
        # Fall through to the large patch if restricted diff fails.

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

    # Install dependencies from V2 install_config before the agent runs.
    # This prevents wasted turns on ImportError / missing test deps.
    # Each instance gets its own PYTHONUSERBASE to prevent dep leakage.
    user_base: str | None = None
    try:
        user_base = await install_deps(instance, workspace)
    except Exception as e:  # noqa: BLE001
        log.warning("install_deps failed for %s: %s", instance_id, e)

    # Build per-instance env vars for run_command (passed via extra_env,
    # NOT os.environ, to avoid race conditions with concurrent instances).
    instance_env: dict[str, str] = {}
    if user_base:
        instance_env["PYTHONUSERBASE"] = user_base
        instance_env["PATH"] = f"{user_base}/bin:{os.environ.get('PATH', '')}"
        instance_env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

    # Build structural index so the agent can use `recall` to locate code.
    if auto_init:
        try:
            import time as _time

            from squishy.index import _build_index_async, save_agents_md
            from squishy.index.store import save_index
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
        from squishy.tools import openai_schemas
        sess = create_session(
            model=model_name,
            working_dir=str(workspace),
            mode="bench",
            tools=openai_schemas("bench"),
        )
        session_id = sess.id
    except Exception:  # noqa: BLE001
        log.debug("session creation failed for %s", instance_id, exc_info=True)

    prompt = build_prompt(instance, workspace=workspace)
    try:
        task_result = await squishy.run(
            prompt, working_dir=str(workspace), timeout=task_timeout,
            session_id=session_id,
            extra_env=instance_env or None,
        )
    except Exception as e:  # noqa: BLE001
        return BenchResult(task_id=instance_id, success=False, error=f"agent: {type(e).__name__}: {e}")

    # Pass agent-edited files so capture_patch can fall back to them
    # if the full diff is bloated with build artifacts.
    edited_files = list(getattr(task_result, "files_edited", []) or [])
    try:
        patch = await capture_patch(
            workspace, instance["base_commit"], edited_files=edited_files,
        )
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

    # Include full conversation transcript for ALL runs (needed for SFT training).
    # Prefer full_log (pre-trim complete history) over messages (post-trim tail).
    full_log = getattr(task_result, "full_log", [])
    diagnostics["transcript"] = full_log if full_log else (
        getattr(task_result, "messages", []) or []
    )

    # Include tool schemas so training data is self-contained.
    try:
        from squishy.tools import openai_schemas
        diagnostics["tool_schemas"] = openai_schemas("bench")
    except Exception:  # noqa: BLE001
        pass

    # Clean up workspace to free disk space and prevent stale state.
    # Remove the entire workspace dir — prepare_workspace() will re-clone
    # if needed on resume. This prevents hundreds of GB of .git dirs from
    # accumulating during full-dataset runs.
    try:
        import shutil
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
    except Exception:  # noqa: BLE001
        log.debug("workspace cleanup failed for %s", instance_id, exc_info=True)

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
    # Prefer full_log (pre-trim complete history) for accurate message-walk counts.
    messages = getattr(task_result, "full_log", []) or getattr(task_result, "messages", []) or []
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


_TB_FILE_RE = re.compile(
    r'File "([^"]+\.py)", line (\d+)',
)


def _extract_traceback_files(text: str) -> list[tuple[str, str]]:
    """Extract (file, line_number) pairs from Python tracebacks in text.

    Returns files in order they appear (bottom of traceback = most relevant).
    Filters out stdlib/site-packages paths.
    """
    matches = _TB_FILE_RE.findall(text)
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for fpath, line_no in matches:
        # Skip stdlib and site-packages
        if any(skip in fpath for skip in ("/lib/python", "site-packages/", "/usr/")):
            continue
        key = f"{fpath}:{line_no}"
        if key not in seen:
            seen.add(key)
            result.append((fpath, line_no))
    return result



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
        proc.returncode,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )