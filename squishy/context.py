"""Project detection, system prompt assembly, conversation history trim.
 
Ported from atlas-proxy/project.go and atlas-proxy/agent.go:buildSystemPrompt.
"""
 
from __future__ import annotations
 
import json
import os
from dataclasses import dataclass
from typing import Any

from squishy.index.store import has_index

MAX_HISTORY = 10  # system + first user + last 8 = 10
 
 
@dataclass
class ProjectInfo:
    language: str = "unknown"
    framework: str = ""
    build_command: str = ""
    test_command: str = ""
    config_files: list[str] | None = None
 
 
def detect_project(cwd: str) -> ProjectInfo:
    files = set(os.listdir(cwd)) if os.path.isdir(cwd) else set()
 
    if "package.json" in files:
        info = ProjectInfo(
            language="javascript",
            build_command="npm run build",
            test_command="npm test",
            config_files=["package.json"],
        )
        try:
            with open(os.path.join(cwd, "package.json")) as f:
                pkg = json.load(f)
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "next" in deps:
                info.framework = "nextjs"
            elif "react" in deps:
                info.framework = "react"
            elif "express" in deps:
                info.framework = "express"
            if "typescript" in deps:
                info.language = "typescript"
        except (OSError, json.JSONDecodeError):
            pass
        return info
 
    if "pyproject.toml" in files or "requirements.txt" in files or "setup.py" in files:
        info = ProjectInfo(
            language="python",
            build_command="python -m build",
            test_command="pytest -q",
            config_files=[f for f in ("pyproject.toml", "requirements.txt", "setup.py") if f in files],
        )
        text = ""
        for fn in info.config_files or []:
            try:
                with open(os.path.join(cwd, fn)) as f:
                    text += f.read()
            except OSError:
                continue
        lowered = text.lower()
        for fw in ("fastapi", "flask", "django"):
            if fw in lowered:
                info.framework = fw
                break
        return info
 
    if "Cargo.toml" in files:
        return ProjectInfo(
            language="rust",
            build_command="cargo build",
            test_command="cargo test",
            config_files=["Cargo.toml"],
        )
 
    if "go.mod" in files:
        return ProjectInfo(
            language="go",
            build_command="go build ./...",
            test_command="go test ./...",
            config_files=["go.mod"],
        )
 
    return ProjectInfo()
 
 
def build_system_prompt(
    cwd: str,
    project: ProjectInfo,
    thinking: bool = False,
    mode: str = "edits",
) -> str:
    files = _top_level_files(cwd)

    thinking_line = "" if thinking else "Do not emit <think> blocks. Be concise.\n"

    project_block = f"Language: {project.language}\n"
    if project.framework:
        project_block += f"Framework: {project.framework}\n"
    if project.build_command:
        project_block += f"Build: {project.build_command}\n"
    if project.test_command:
        project_block += f"Test: {project.test_command}\n"

    index_block = _index_header(cwd)
    instructions_block = load_agent_instructions(cwd)
    mode_block = _mode_block(mode, cwd)
    recall_rule = _recall_rule(cwd)

    return f"""You are squishy, a local coding assistant that edits files and runs commands to complete the user's task.

{thinking_line}
## Rules
- Read files before editing them.
- For existing files longer than 100 lines, always use `edit_file` (not `write_file`).
- Use relative paths. Working directory is already set.
- Verify your work with `run_command` (run the tests or the program itself) after making changes.
- Explore thoroughly when fixing bugs or implementing features - it's better to understand the codebase than to guess.
- When you finish the user's task, respond with plain text summarizing what you did (no tool call).
- Do not re-read a file you have already read in this conversation unless you need a different line range. Use what you have.
{recall_rule}

## File References
- Users can reference files in their input using `@filename` syntax.
- When a user includes `@some/path.py`, the full file contents are automatically injected into the conversation wrapped in `<file>` tags.
- File content is labeled with path and line count so you know exactly what file it is.

## Planning
- For complex tasks, call `plan_task` first to present a structured plan with problem, solution, steps, and files.
- `plan_task` is valid as soon as you have enough information to propose a solid approach — do not wait for exhaustive research.
- `files_to_modify` and `files_to_create` may be partial or empty when some file choices are still uncertain.
- The user will be asked to approve the plan before you proceed.
- After the plan is approved, call `update_plan(step_index=N, status="done")` as you complete each step.
- This keeps the user informed of progress through their task.

{mode_block}
## Project
{project_block}
## Working directory
{cwd}

## Top-level files
{', '.join(files) if files else '(empty)'}
{index_block}{instructions_block}"""
 
 
_INSTRUCTION_SOURCES: tuple[tuple[str, str], ...] = (
    ("AGENTS.md", "AGENTS.md"),
    ("CLAUDE.md", "CLAUDE.md"),
    ("SQUISHY.md", "SQUISHY.md"),
    (".squishy/AGENTS.md", ".squishy/AGENTS.md"),
)
_INSTRUCTION_CAP_BYTES = 4096


def load_agent_instructions(cwd: str) -> str:
    """Load project-local agent instructions into a system-prompt block.

    Checks (in order) AGENTS.md, CLAUDE.md, SQUISHY.md at the repo root and
    .squishy/AGENTS.md (auto-generated by `/init`). Each file found is wrapped
    in its own section and capped at ~4 KB to bound context cost.
    """
    if not os.path.isdir(cwd):
        return ""
    parts: list[str] = []
    for rel, label in _INSTRUCTION_SOURCES:
        path = os.path.join(cwd, rel)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                body = f.read(_INSTRUCTION_CAP_BYTES + 1)
        except OSError:
            continue
        if not body.strip():
            continue
        if len(body) > _INSTRUCTION_CAP_BYTES:
            body = body[:_INSTRUCTION_CAP_BYTES] + "\n…(truncated)"
        parts.append(f"\n## Agent instructions ({label})\n{body.rstrip()}\n")
    return "".join(parts)


def _recall_rule(cwd: str) -> str:
    """Return the 'use recall first' rule, strengthened when an index exists."""
    index_path = os.path.join(cwd, ".squishy", "index.json")
    if os.path.isfile(index_path):
        return (
            "- An index is present at `.squishy/index.json`. Before calling "
            "`read_file` or `list_directory`, call `recall(query=...)` at least "
            "once per new topic to locate the right files. Do not read files blindly."
        )
    return (
        "- To locate files or symbols, prefer `recall(query=...)` over walking with "
        "`list_directory` when an index is present."
    )


def _mode_block(mode: str, cwd: str) -> str:
    index_available = has_index(cwd)
    if mode == "plan":
        index_guidance = (
            "- **In plan mode, your FIRST tool call should ALWAYS be `recall(query=...)` to use the index.**\n"
            "- After `recall`, make 1-2 targeted reads to understand the problem.\n"
            "- Call `plan_task` within your first 2-3 turns. Usually: recall → targeted reads → plan.\n"
            "- Do NOT call read_file, list_directory, or search_files without first using `recall`. The index exists for efficient navigation.\n"
            "- If the user already named likely files, use `recall` first to verify location, then inspect directly.\n"
            "- Example workflow in plan mode:\n"
            '  1. `recall(query="function name or file pattern")`\n'
            '  2. `read_file(path="relevant_file.py", offset=..., limit=...)`\n'
            '  3. `plan_task(problem="...", solution="...", steps=["..."])`\n'
        ) if index_available else (
            "- No repo index is present yet, so you may use targeted `read_file`, `list_directory`, or `search_files` calls to investigate.\n"
            "- Prefer 1-3 focused reads, then call `plan_task`; do not wait for exhaustive research.\n"
            "- If navigation is difficult, ask the user to run `/init` or use it yourself once you are allowed to leave plan mode.\n"
            "- Example workflow in plan mode without an index:\n"
            '  1. `list_directory(path=".")`\n'
            '  2. `read_file(path="relevant_file.py", offset=..., limit=...)`\n'
            '  3. `plan_task(problem="...", solution="...", steps=["..."])`\n'
        )
        return (
            "## Mode: plan (read-only)\n"
            "- **CRITICAL: For ANY task requiring file changes, call `plan_task` FIRST.**\n"
            "- Do NOT attempt implementation until after the plan is approved.\n"
            "- For simple tasks (e.g., reading one file), you may skip plan_task.\n"
            f"{index_guidance}"
            "- Call `plan_task` as soon as you can explain the problem, solution, and concrete steps. `files_to_modify`/`files_to_create` may be partial or empty if uncertain.\n"
            "- Do NOT end the turn with prose before calling `plan_task`.\n"
            '  ```json\n'
            '  {\n'
            '    "problem": "What needs to be fixed or implemented",\n'
            '    "solution": "High-level approach to solve it",\n'
            '    "steps": ["Step 1 description", "Step 2 description"],\n'
            '    "files_to_modify": ["file1.py", "file2.py"],\n'
            '    "files_to_create": ["new_file.py"]\n'
            '  }\n'
            '  ```\n'
            "- `run_command` is limited to read-only commands: ls, cat, head, tail, wc, grep, rg, find, "
            "pwd, which, file, stat, tree, ruff check, mypy, pyright, git status/log/diff/show/branch/blame/ls-files, "
            "pytest --collect-only, python -m pytest --collect-only. No pipes, redirects, or command chains.\n"
            "- After the user approves the plan, they will switch you into edits mode to execute it.\n"
        )
    if mode == "yolo":
        return (
            "## Mode: yolo\n"
            "- All tools available without approval prompts. Be careful.\n"
            "- **For non-trivial tasks, call `plan_task` first to structure your approach.**\n"
        )
    return (
        "## Mode: edits\n"
        "- `run_command` requires user approval on each call.\n"
        "- **For non-trivial tasks, call `plan_task` first to present a structured plan.**\n"
        "- If the user approved a plan, follow it: after each step call "
        "`update_plan(step_index=N, status=\"done\")`.\n"
    )


def _index_header(cwd: str) -> str:
    """Return a compact, ~200-token block summarizing the cached repo index.
 
    Silently returns empty string when no `.squishy/index.json` exists.
    """
    try:
        from squishy.index.store import load_index, load_meta
    except Exception:  # noqa: BLE001
        return ""
    try:
        meta = load_meta(cwd)
        idx = load_index(cwd)
    except Exception:  # noqa: BLE001
        return ""
    if meta is None or idx is None:
        return ""
 
    stats = meta.stats or {}
    by_ext = sorted(
        ((k.removeprefix("ext"), v) for k, v in stats.items() if k.startswith("ext")),
        key=lambda kv: -kv[1],
    )[:6]
    ext_line = ", ".join(f"{v} {k or '?'}" for k, v in by_ext) if by_ext else "?"
 
    # Top-N directories by descendant file count.
    dir_counts: list[tuple[str, int]] = []
    for node in idx.root.walk():
        if node.kind == "dir" and node.path:
            n = sum(1 for c in node.walk() if c.kind == "file")
            dir_counts.append((node.path, n))
    dir_counts.sort(key=lambda kv: -kv[1])
    top_dirs = ", ".join(f"{p} ({n})" for p, n in dir_counts[:5]) or "(flat)"
 
    import time as _t
    age_s = max(0.0, _t.time() - (meta.generated_at or 0.0))
    if age_s < 120:
        age = f"{int(age_s)}s"
    elif age_s < 7200:
        age = f"{int(age_s / 60)}m"
    else:
        age = f"{int(age_s / 3600)}h"
 
    return (
        "\n## Repo index\n"
        f"{stats.get('files', 0)} files, {stats.get('symbols', 0)} symbols. "
        f"By ext: {ext_line}. "
        f"Top dirs: {top_dirs}. "
        f"Indexed {age} ago. Use `recall(query=...)` to navigate.\n"
    )
 
 
def _top_level_files(cwd: str, limit: int = 50) -> list[str]:
    if not os.path.isdir(cwd):
        return []
    from squishy.tools.fs import SKIP_DIRS
    out = []
    for name in sorted(os.listdir(cwd)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out
 
 
def trim_history(messages: list[dict[str, Any]], max_messages: int = MAX_HISTORY) -> list[dict[str, Any]]:
    """Keep system + first user + last (max_messages - 2) messages.

    Ported from atlas-proxy/agent.go:41-50. Preserves initial intent while
    bounding context size.

    Pair-aware: the tail is never allowed to begin with a ``role="tool"``
    message, because an orphan tool result whose matching assistant
    ``tool_calls`` has been sliced off will confuse the LLM (it sees a tool
    result it has no record of requesting, and re-requests the same read).
    Leading tool messages are dropped until we hit an assistant or user turn.
    """
    if len(messages) <= max_messages:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        return system

    first_user_idx = next((i for i, m in enumerate(non_system) if m.get("role") == "user"), 0)
    first_user = [non_system[first_user_idx]]
    remaining_budget = max_messages - len(system) - len(first_user)
    tail = non_system[-remaining_budget:] if remaining_budget > 0 else []
    if tail and tail[0] is first_user[0]:
        tail = tail[1:]

    # Drop leading orphan tool results. Their matching assistant tool_calls
    # message has been trimmed away, so the LLM can't associate the result
    # with a prior action.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]

    return system + first_user + tail
