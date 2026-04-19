"""Project detection, system prompt assembly, conversation history trim.
 
Ported from atlas-proxy/project.go and atlas-proxy/agent.go:buildSystemPrompt.
"""
 
from __future__ import annotations
 
import json
import os
from dataclasses import dataclass
from typing import Any
 
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
 
 
def build_system_prompt(cwd: str, project: ProjectInfo, thinking: bool = False) -> str:
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
 
    return f"""You are squishy, a local coding assistant that edits files and runs commands to complete the user's task.

{thinking_line}
## Rules
- Read files before editing them.
- For existing files longer than 100 lines, always use `edit_file` (not `write_file`).
- Use relative paths. Working directory is already set.
- Verify your work with `run_command` (run the tests or the program itself) after making changes.
- Explore thoroughly when fixing bugs or implementing features - it's better to understand the codebase than to guess.
- When you finish the user's task, respond with plain text summarizing what you did (no tool call).
- To locate files or symbols, prefer `recall(query=...)` over walking with `list_directory` when an index is present.

## Project
{project_block}
## Working directory
{cwd}

## Top-level files
{', '.join(files) if files else '(empty)'}
{index_block}"""
 
 
def _index_header(cwd: str) -> str:
    """Return a compact, ~200-token block summarizing the cached repo index.
 
    Silently returns empty string when no `.squishy/index.json` exists.
    """
    try:
        from squishy.index.store import load_index, load_meta
    except Exception:  # noqa: BLE001
        return ""
    meta = load_meta(cwd)
    idx = load_index(cwd)
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
    return system + first_user + tail