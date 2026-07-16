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
import ast
import json as _json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from squishy.api import Squishy
from squishy.agent_state import distinct_f2p_files
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


async def install_deps(
    instance: dict[str, Any], workspace: Path,
) -> tuple[str | None, dict[str, Any]]:
    """Run install_config.install commands in an isolated prefix.

    V2 instances carry ``install_config.install`` — a list of shell commands
    (usually pip install) that must run before tests work.  Running these
    *before* the agent starts prevents wasted turns on ImportError /
    ModuleNotFoundError failures.

    Each instance gets its own ``PYTHONUSERBASE`` directory so packages from
    one instance cannot leak into another.  The caller must set
    ``os.environ["PYTHONUSERBASE"]`` to the returned path (and add its
    ``bin/`` to ``PATH``) before launching the agent, then clean up after.

    Returns ``(user_base, install_status)`` where ``install_status`` is::

        {
            "ok": bool,                     # True iff every command succeeded
            "reason": str | None,           # short summary if not ok
            "timed_out_cmds": list[str],    # commands that hit 300s timeout
            "failed_cmds": list[str],       # commands that returned non-zero
            "ran": int,                     # how many commands attempted
        }

    ``user_base`` is None when there are no install commands.
    """
    install_status: dict[str, Any] = {
        "ok": True,
        "reason": None,
        "timed_out_cmds": [],
        "failed_cmds": [],
        "ran": 0,
    }
    install_config = instance.get("install_config") or {}
    cmds: list[str] = install_config.get("install") or []
    if not cmds:
        return None, install_status

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
        install_status["ran"] += 1
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
            install_status["ok"] = False
            install_status["timed_out_cmds"].append(cmd[:120])
            install_status["reason"] = "timeout"
            # Clean up leaked .pth files even on timeout.
            for d, snap in pth_snapshots.items():
                _cleanup_leaked_pth(d, snap)
            return user_base, install_status

        # Clean up any .pth files that leaked into system site-packages.
        for d, snap in pth_snapshots.items():
            _cleanup_leaked_pth(d, snap)

        if proc.returncode != 0:
            err = stderr_b.decode("utf-8", errors="replace")[:200]
            log.warning(
                "install failed for %s (rc=%d): %s → %s",
                instance["instance_id"], proc.returncode, cmd[:80], err,
            )
            install_status["ok"] = False
            install_status["failed_cmds"].append(cmd[:120])
            if install_status["reason"] is None:
                install_status["reason"] = "non-zero exit"
            # Continue — some commands may fail non-fatally (e.g. optional extras)
    return user_base, install_status
 
 
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


def _extract_failing_test_imports(
    workspace: str | Path, fail_to_pass: list[str],
) -> dict[str, list[str]]:
    """For each FAIL_TO_PASS test path, parse it with ast and return the
    repo-internal modules it imports.

    Returns ``{test_path: [module1, module2, ...]}``. Skips non-Python files
    silently. Caps at 5 imports per test, dedupes.
    """
    workspace = Path(workspace)
    if not workspace.exists():
        return {}

    # Heuristic: top-level package names = subdirs that contain an __init__.py
    # OR top-level .py files, excluding test/build/venv dirs.
    repo_pkgs: set[str] = set()
    try:
        for entry in workspace.iterdir():
            name = entry.name
            if name.startswith(".") or name in {
                "tests", "test", "build", "dist", "venv", ".venv",
                "node_modules", "__pycache__",
            }:
                continue
            if entry.is_dir() and (entry / "__init__.py").exists():
                repo_pkgs.add(name)
            elif entry.is_file() and name.endswith(".py"):
                repo_pkgs.add(name[:-3])
    except OSError:
        return {}

    if not repo_pkgs:
        return {}

    out: dict[str, list[str]] = {}
    for test_id in fail_to_pass:
        test_path = test_id.split("::", 1)[0] if "::" in test_id else test_id
        if not test_path.endswith(".py"):
            continue
        if test_path in out:
            continue
        full = workspace / test_path
        if not full.exists():
            continue
        try:
            src = full.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError):
            continue

        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Relative imports point at sibling modules in the same package.
                if node.level and node.level > 0:
                    rel = "." * node.level + (node.module or "")
                    if rel and rel not in modules:
                        modules.append(rel)
                elif node.module:
                    top = node.module.split(".", 1)[0]
                    if top in repo_pkgs and node.module not in modules:
                        modules.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in repo_pkgs and alias.name not in modules:
                        modules.append(alias.name)
        if modules:
            out[test_path] = modules[:5]
    return out


def _common_prefix_len(a: str, b: str) -> int:
    """Length of the longest shared character prefix of *a* and *b*."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _extract_test_context_bodies(
    workspace: str | Path,
    fail_to_pass: list[str],
    *,
    max_total_lines: int = 60,
    max_per_test_lines: int = 25,
    max_sibling_per_file: int = 2,
) -> list[tuple[str, str, bool]]:
    """For each FAIL_TO_PASS test ID, return (label, source, is_sibling).

    For SWE-bench, the F2P test function is added by ``test_patch`` at eval
    time and does NOT exist in the workspace — the harness sees only the
    pre-patch tree.  Falling back to sibling tests in the same file gives
    the agent the *test conventions* (helper fixtures, expected return
    shapes, parametrize style) without needing the exact target body.

    Algorithm:
    1. Resolve the test_id to a file path.
    2. AST-parse the file. If the F2P function exists → use that body
       (label=test_id, is_sibling=False).
    3. Else, pick up to ``max_sibling_per_file`` sibling test functions
       whose names share the longest prefix with the F2P function name
       (label="<test_path>::<sibling_name>  # sibling of <f2p_func>",
       is_sibling=True).

    Caps total lines globally and per-test.  Returns [] when nothing
    useful can be extracted.
    """
    workspace = Path(workspace)
    if not workspace.exists():
        return []

    out: list[tuple[str, str, bool]] = []
    total_lines = 0
    seen_funcs: set[tuple[str, str]] = set()  # (path, name)

    def _emit_body(test_path: str, label: str, target: ast.AST, src: str,
                   is_sibling: bool) -> int:
        """Append a body to *out*; return number of lines emitted."""
        nonlocal total_lines
        src_lines = src.splitlines()
        start = max(0, target.lineno - 1)
        end = getattr(target, "end_lineno", None)
        if end is None:
            end = min(len(src_lines), start + max_per_test_lines)
        body_lines = src_lines[start:end]
        if len(body_lines) > max_per_test_lines:
            body_lines = body_lines[:max_per_test_lines] + [
                f"    # ... ({len(src_lines[start:end]) - max_per_test_lines} more lines truncated)"
            ]
        remaining = max_total_lines - total_lines
        if remaining <= 0:
            return 0
        if len(body_lines) > remaining:
            body_lines = body_lines[:remaining]
        out.append((label, "\n".join(body_lines), is_sibling))
        total_lines += len(body_lines)
        return len(body_lines)

    for test_id in fail_to_pass:
        if total_lines >= max_total_lines:
            break
        if "::" not in test_id:
            continue
        parts = test_id.split("::")
        test_path = parts[0]
        func_name = parts[-1].split("[", 1)[0].strip()
        if not func_name or not test_path.endswith(".py"):
            continue
        full = workspace / test_path
        if not full.exists():
            continue
        try:
            src = full.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError):
            continue

        # Collect every top-level/method test function in the file.
        all_funcs: list[tuple[str, ast.AST]] = []
        target_func: ast.AST | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name and target_func is None:
                    target_func = node
                if node.name.startswith("test_") or node.name.startswith("Test"):
                    all_funcs.append((node.name, node))

        if target_func is not None:
            key = (test_path, func_name)
            if key in seen_funcs:
                continue
            seen_funcs.add(key)
            _emit_body(test_path, test_id, target_func, src, is_sibling=False)
            continue

        # SWE-bench fallback: F2P function isn't in the workspace.  Pick the
        # siblings whose names share the longest prefix with the F2P name.
        candidates = [
            (name, node) for name, node in all_funcs if name != func_name
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda nn: (-_common_prefix_len(nn[0], func_name), nn[0])
        )
        emitted = 0
        for name, node in candidates:
            if emitted >= max_sibling_per_file:
                break
            if total_lines >= max_total_lines:
                break
            key = (test_path, name)
            if key in seen_funcs:
                continue
            seen_funcs.add(key)
            label = f"{test_path}::{name}  # sibling of {func_name}"
            if _emit_body(test_path, label, node, src, is_sibling=True) > 0:
                emitted += 1
    return out


# recall_from_index now lives in squishy.tools.recall (so the core loop never
# imports from squishy.bench). Re-exported here under the historical name for
# the bench call sites and existing tests.
from squishy.tools.recall import recall_from_index as _recall_from_index


def build_prompt(
    instance: dict[str, Any],
    *,
    workspace: str | Path | None = None,
    install_status: dict[str, Any] | None = None,
) -> str:
    """Compose a user prompt from the SWE-bench instance fields.

    Workflow instructions live in the system prompt (via ``_mode_block("bench")``
    in context.py). This user message contains ONLY the problem statement,
    failing tests, hints, and (optionally) index-based code pointers — keeping
    it small so it survives trimming.

    ``install_status`` (optional) — when present and ``ok=False``, prepends
    an environment-degradation warning so the agent knows tests may not run.
    """
    parts: list[str] = []

    # Surface install-deps degradation up front — agent should focus on the
    # source edit and not waste turns trying to run a broken pytest.
    if install_status and not install_status.get("ok", True):
        reason = install_status.get("reason") or "unknown"
        timed_out = install_status.get("timed_out_cmds") or []
        failed = install_status.get("failed_cmds") or []
        parts += [
            "## ⚠ Environment Setup Warning",
            f"Dependency install was degraded (reason: {reason}).",
            "Tests may not run successfully — focus on producing a syntactically",
            "valid patch and do NOT assume `pytest` or `python -m {pkg}` will work.",
        ]
        if timed_out:
            parts.append(f"- timed out commands: {len(timed_out)}")
        if failed:
            parts.append(f"- failed commands: {len(failed)}")
        parts.append("")

    parts += [
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
        # Extract unique test files from "path/to/test.py::test_name" identifiers
        # and tell the agent to read them. The new failing tests typically follow
        # the conventions of existing tests in the same file — reading them up
        # front prevents wrong-fix patches that miss output-format expectations.
        test_files: list[str] = []
        for test in fail_to_pass:
            if "::" in test:
                tf = test.split("::", 1)[0]
                if tf and tf not in test_files:
                    test_files.append(tf)
        if test_files:
            parts.append("")
            files_list = ", ".join(f"`{tf}`" for tf in test_files[:3])
            parts.append(
                f"Before editing source, `read_file` on the test file(s) "
                f"({files_list}) to learn the existing test conventions — "
                f"the new failing tests will follow the same patterns "
                f"(output format, helper fixtures, expected return shapes)."
            )

        # When FAIL_TO_PASS spans multiple test files, surface the grouping
        # explicitly.  Generic, repo-agnostic: comes purely from the harness
        # data.  Symptom this prevents: agent fixes the bug exercised by
        # one F2P file, runs only that file, sees green, calls finish_plan
        # and ships — leaving the second file's regression untouched
        # (the v27.x scico-561 failure mode: 2D test fixed, 3D test ignored).
        f2p_files_set = distinct_f2p_files(fail_to_pass)
        if len(f2p_files_set) >= 2:
            grouped: dict[str, list[str]] = {}
            for tid in fail_to_pass:
                if "::" not in tid:
                    continue
                tf = tid.split("::", 1)[0].replace("\\", "/").strip()
                if tf:
                    grouped.setdefault(tf, []).append(tid)
            parts += [
                "",
                f"## Failing Tests Span {len(f2p_files_set)} Files",
                "Each file below has its own failing tests.  Read EACH file,",
                "and make sure your fix addresses the bug class in ALL of them",
                "— a fix that only makes one file pass is incomplete.",
            ]
            for tf in list(grouped.keys())[:5]:
                tids = grouped[tf]
                preview = ", ".join(f"`{t.split('::', 1)[1]}`" for t in tids[:2])
                more = f" (+{len(tids) - 2} more)" if len(tids) > 2 else ""
                parts.append(f"- `{tf}` — {len(tids)} test(s): {preview}{more}")
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

    # Tell the agent which repo-internal modules each failing test imports.
    # This anchors the search to the *right code path* — both v25 home-assistant
    # patches edited the wrong files because the agent never looked at where
    # the failing tests pointed.
    if workspace and fail_to_pass:
        try:
            test_imports = _extract_failing_test_imports(workspace, fail_to_pass)
        except Exception:  # noqa: BLE001
            test_imports = {}
        if test_imports:
            parts += ["", "## Where the failing tests are looking"]
            seen_modules: set[str] = set()
            for tpath, modules in list(test_imports.items())[:5]:
                deduped = [m for m in modules if not (m in seen_modules or seen_modules.add(m))]
                if not deduped:
                    continue
                joined = ", ".join(f"`{m}`" for m in deduped[:5])
                parts.append(f"- `{tpath}` imports: {joined}")
            parts.append(
                "Read these modules first — the bug is most likely in one of them."
            )

        # Inject test source bodies — either the real F2P body (when the
        # function exists in-tree) or sibling tests in the same file (the
        # SWE-bench case, where the F2P function is added by test_patch
        # at eval time).  Both cases reveal the test conventions (helpers,
        # parametrize style, return shapes) which are the highest-signal
        # info for "edit the right file."
        try:
            test_bodies = _extract_test_context_bodies(workspace, fail_to_pass)
        except Exception:  # noqa: BLE001
            test_bodies = []
        if test_bodies:
            any_sibling = any(is_sib for _, _, is_sib in test_bodies)
            if any_sibling:
                parts += [
                    "",
                    "## Test File Context",
                    "The actual failing tests are added at evaluation time and",
                    "are NOT in your workspace yet.  The bodies below are SIBLING",
                    "tests from the same file — use them to learn the test",
                    "conventions (helper fixtures, parametrize style, expected",
                    "return shapes) that the real failing tests will follow.",
                ]
            else:
                parts += [
                    "",
                    "## Failing Test Bodies",
                    "These are the test functions you must make pass.  Match the",
                    "function names, argument shapes, and return values they",
                    "exercise.",
                ]
            for label, body, _is_sibling in test_bodies:
                parts.append(f"\n### `{label}`")
                parts.append("```python")
                parts.append(body)
                parts.append("```")

    # Pre-query the index to give the agent code pointers.
    # Include failing test paths in the query so the index surfaces the
    # functions that the tests actually call, not just keywords from the
    # problem statement.
    if workspace:
        problem_text = instance.get("problem_statement", "")
        # Augment query with failing test names — these contain the modules
        # and functions being tested (e.g. "tests/test_utils.py::test_clip_boxes"
        # tells the index to surface "clip_boxes" from the source).
        query_parts = [problem_text]
        for test_id in fail_to_pass[:5]:
            # Extract function/class names from test IDs
            for part in test_id.replace("::", " ").replace("/", " ").replace(".", " ").split():
                if part.startswith("test_"):
                    # "test_clip_boxes" → "clip_boxes"
                    query_parts.append(part[5:])
                elif part.startswith("Test"):
                    query_parts.append(part[4:])
        index_query = " ".join(query_parts)
        pointers = _recall_from_index(workspace, index_query, limit=8)
        if pointers:
            parts += ["", "## Relevant Code (from index)"]
            # Detect sibling-class clusters generically: any path that has
            # ≥2 class entries.  When present, lead with a one-line nudge
            # so the agent doesn't fix one class and miss its siblings
            # (the v27.x scico-561 failure mode — XRayTransform2D fixed,
            # XRayTransform3D in the same file ignored).  Generic: the
            # detection is purely structural, no repo-specific names.
            class_paths: dict[str, int] = {}
            for p in pointers:
                if p.get("kind") == "class":
                    class_paths[p["path"]] = class_paths.get(p["path"], 0) + 1
            sibling_paths = sorted(p for p, n in class_paths.items() if n >= 2)
            if sibling_paths:
                preview = ", ".join(f"`{p}`" for p in sibling_paths[:2])
                more = f" (+{len(sibling_paths) - 2} more)" if len(sibling_paths) > 2 else ""
                parts.append(
                    f"NOTE: multiple classes from the same file are listed "
                    f"below ({preview}{more}).  Classes that share a file "
                    f"often share a bug — when you fix one, check whether "
                    f"its siblings need the same fix."
                )
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


def _build_retry_prompt(instance: dict[str, Any], *, workspace: str | Path | None = None) -> str:
    """Build a focused retry prompt for 0-patch runs.

    Much shorter and more directive than the full prompt — tells the model
    to read ONE file and make ONE edit, no exploration.
    """
    problem = instance.get("problem_statement", "").strip()
    if len(problem) > 1500:
        problem = problem[:1500] + "\n..."

    parts = [
        "A previous attempt to fix this bug ran out of time without making any edits.",
        "You MUST fix this in as few turns as possible. DO NOT explore — go straight to editing.",
        "",
        "## Problem",
        problem,
    ]

    if workspace:
        pointers = _recall_from_index(workspace, problem, limit=5)
        if pointers:
            parts += ["", "## Most Relevant Files"]
            for p in pointers[:3]:
                line_info = ""
                if p.get("lines"):
                    line_info = f" (L{p['lines'][0]}-{p['lines'][1]})"
                parts.append(f"- `{p['path']}`{line_info}: `{p['name']}`")

    parts += [
        "",
        "## Instructions",
        "1. `read_file` on the most relevant file above.",
        "2. `edit_file` with your fix. Do this IMMEDIATELY after reading.",
        "3. You are done. Do NOT run tests or explore further.",
        "",
        "Make your BEST GUESS fix now. Any fix is better than no fix.",
    ]
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
    # pbr / setuptools-generated metadata files (modified by `pip install -e .`)
    ":(exclude)AUTHORS",
    ":(exclude)ChangeLog",
    ":(exclude)PKG-INFO",
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

    # Prefer a restricted diff (agent-edited files only) when the full diff
    # contains changes the agent didn't make (e.g. pbr-generated metadata,
    # build artifacts modified by `pip install -e .`).
    if edited_files:
        rc2, out2, err2 = await _git(
            ["diff", "--cached", "--no-color", base_commit, "--"]
            + list(edited_files),
            cwd=workspace,
        )
        if rc2 == 0 and out2.strip():
            if out2 != out:
                log.info(
                    "using restricted diff (%d edited files, %d bytes vs %d full)",
                    len(edited_files), len(out2.encode()), len(out.encode()),
                )
            return out2

    # If the full diff is excessively large and we have no edited_files,
    # return empty to avoid polluted patches.
    if len(out.encode()) > _MAX_PATCH_BYTES:
        log.warning("patch too large (%d bytes) with no edited files — returning empty", len(out.encode()))
        return ""

    # Heuristic: if the patch creates many new files and the agent tracked
    # no edits, it's almost certainly install/test side-effect pollution
    # (e.g. biomass copy_to_current dumping a model package into cwd).
    if not edited_files:
        new_file_count = sum(1 for ln in out.split("\n") if ln.startswith("new file mode "))
        if new_file_count >= 10:
            log.warning(
                "patch creates %d new files with no tracked edits — likely "
                "side-effect pollution; returning empty", new_file_count,
            )
            return ""

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
    t0 = time.monotonic()
    try:
        workspace = await prepare_workspace(instance, workspace_root)
    except BenchError as e:
        return BenchResult(
            task_id=instance_id, success=False, error=f"workspace: {e}",
            elapsed_s=time.monotonic() - t0,
        )

    # Install dependencies from V2 install_config before the agent runs.
    # This prevents wasted turns on ImportError / missing test deps.
    # Each instance gets its own PYTHONUSERBASE to prevent dep leakage.
    user_base: str | None = None
    install_status: dict[str, Any] = {"ok": True, "reason": None}
    try:
        user_base, install_status = await install_deps(instance, workspace)
    except Exception as e:  # noqa: BLE001
        log.warning("install_deps failed for %s: %s", instance_id, e)
        install_status = {
            "ok": False, "reason": f"exception: {type(e).__name__}",
            "timed_out_cmds": [], "failed_cmds": [], "ran": 0,
        }

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

    prompt = build_prompt(instance, workspace=workspace, install_status=install_status)

    # Thread eval-time metadata to the agent loop via ToolContext.notes.
    # `inject_test_failure_nudge` consults `fail_to_pass_tests` to filter and
    # prioritize F2P failures over noisy collection errors.  `install_status`
    # tells goal-drift detection to suppress "fix imports" nudges when the
    # environment was already broken before the agent started.
    fail_to_pass_for_notes = instance.get("FAIL_TO_PASS") or []
    if isinstance(fail_to_pass_for_notes, str):
        try:
            fail_to_pass_for_notes = _json.loads(fail_to_pass_for_notes)
        except (ValueError, TypeError):
            fail_to_pass_for_notes = (
                [fail_to_pass_for_notes] if fail_to_pass_for_notes.strip() else []
            )
    agent_notes: dict[str, str] = {}
    if fail_to_pass_for_notes:
        agent_notes["fail_to_pass_tests"] = _json.dumps(fail_to_pass_for_notes)
    agent_notes["install_status"] = _json.dumps(install_status)
    # v6e: forward V2's install_config.test_cmd so the post-edit nudge can
    # suggest the right runner (e.g. "npm test", "./vendor/bin/phpunit")
    # instead of hardcoded pytest.  Absent on V1; harmless empty string then.
    test_cmd_for_notes = (instance.get("install_config") or {}).get("test_cmd", "")
    if isinstance(test_cmd_for_notes, str) and test_cmd_for_notes.strip():
        agent_notes["test_cmd"] = test_cmd_for_notes.strip()

    agent_error: str | None = None
    task_result = None
    try:
        task_result = await squishy.run(
            prompt, working_dir=str(workspace), timeout=task_timeout,
            session_id=session_id,
            extra_env=instance_env or None,
            notes=agent_notes or None,
        )
    except Exception as e:  # noqa: BLE001
        agent_error = f"agent: {type(e).__name__}: {e}"
        log.warning("agent error for %s: %s", instance_id, agent_error)
        # Recover the partial TaskResult attached by Agent.run() on
        # AgentTimeout/AgentCancelled so the transcript and turn_log
        # accumulated up to the failure are not lost.  Without this the
        # bench prediction has empty diagnostics whenever the wall-clock
        # timeout fires (which is most "hard" instances).
        partial = getattr(e, "partial_result", None)
        if partial is not None:
            task_result = partial

    # Always attempt to capture the patch — even on timeout or error,
    # the agent may have made valid edits in the workspace.
    edited_files = list(getattr(task_result, "files_edited", []) or []) if task_result else []
    try:
        patch = await capture_patch(
            workspace, instance["base_commit"], edited_files=edited_files,
        )
    except BenchError as e:
        patch = ""
        log.warning("capture_patch failed for %s: %s", instance_id, e)

    # Retry once if the first attempt produced 0 patch. Use a focused
    # prompt and shorter timeout — any edit is better than no edit.
    if not patch.strip():
        log.info("0-patch for %s — attempting retry with focused prompt", instance_id)
        # Reset workspace to base_commit so the retry starts clean.
        await _git(["checkout", "-f", instance["base_commit"]], cwd=workspace)
        await _git(["clean", "-fdx"], cwd=workspace)
        retry_prompt = _build_retry_prompt(instance, workspace=workspace)
        retry_result = None
        try:
            retry_result = await squishy.run(
                retry_prompt, working_dir=str(workspace),
                timeout=min(task_timeout, 600),  # shorter timeout for retry
                extra_env=instance_env or None,
                notes=agent_notes or None,
            )
        except Exception as e2:  # noqa: BLE001
            log.warning("retry also failed for %s: %s", instance_id, e2)
            if not agent_error:
                agent_error = f"retry: {type(e2).__name__}: {e2}"
            partial2 = getattr(e2, "partial_result", None)
            if partial2 is not None:
                retry_result = partial2
        retry_edited = list(getattr(retry_result, "files_edited", []) or []) if retry_result else []
        try:
            retry_patch = await capture_patch(
                workspace, instance["base_commit"], edited_files=retry_edited,
            )
            if retry_patch.strip():
                log.info("retry produced %d-byte patch for %s", len(retry_patch), instance_id)
                patch = retry_patch
                task_result = retry_result
                edited_files = retry_edited
                agent_error = None  # Clear error since retry succeeded
        except BenchError:
            pass

    if not patch.strip() and agent_error:
        return BenchResult(
            task_id=instance_id, success=False, error=agent_error,
            elapsed_s=time.monotonic() - t0,
        )
 
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }
    # A non-empty patch counts as success even if the agent hit max turns,
    # because the fix was applied — the agent just didn't stop cleanly.
    has_patch = bool(patch.strip())

    # Collect diagnostics from the task result for post-hoc analysis.
    diagnostics: dict[str, Any] = {}
    if task_result is not None:
        diagnostics = _extract_diagnostics(task_result)
    if agent_error:
        diagnostics["agent_error"] = agent_error
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
        error="" if has_patch else (
            (task_result.error if task_result else agent_error) or "empty patch"
        ),
        elapsed_s=(
            task_result.elapsed_s if task_result and task_result.elapsed_s
            else time.monotonic() - t0
        ),
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