"""Project detection, system prompt assembly, conversation history trim.
 
Ported from atlas-proxy/project.go and atlas-proxy/agent.go:buildSystemPrompt.
"""
 
from __future__ import annotations
 
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from squishy.index.store import has_index
from squishy.tools.fs import SKIP_DIRS

 
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
    """Assemble the system prompt.

    Goals (after the rewrite):
      * Each rule appears once. Recall, planning, and "use update_plan
        after each step" used to be repeated across `## Rules`,
        `## Planning`, and the per-mode block — now each lives in
        exactly one place.
      * No verbose example blocks (JSON shape, run_command allowlists)
        in prose: the tool schemas already carry that content, and
        runtime errors echo allowlists when the model gets them wrong.
      * Single-line summary blocks (project, index, top-level files)
        replace the previous 4-6 line versions.
    """
    thinking_line = "" if thinking else "Do not emit <think> blocks. Be concise.\n"

    has_idx = has_index(cwd)
    rules = _rules_block(has_idx)
    mode_block = _mode_block(mode, cwd)
    project_line = _project_line(project)
    index_block = _index_header(cwd)
    top_files_block = "" if has_idx else _top_level_files_block(cwd)
    mcp_block = _mcp_block()
    instructions_block = load_agent_instructions(cwd)

    parts = [
        "You are squishy, a local coding assistant that edits files and runs commands to complete the user's task.",
        "",
        thinking_line.rstrip(),
        rules,
        mode_block,
        f"## Project\n{project_line}",
        f"## Working dir\n{cwd}",
    ]
    # Drop empty pieces (thinking_line collapses to "" when thinking is on).
    parts = [p for p in parts if p]
    body = "\n\n".join(parts)
    # Tail blocks already start with their own leading "\n" or are empty.
    return body + index_block + top_files_block + mcp_block + instructions_block


def _rules_block(has_idx: bool) -> str:
    """Core rules. Recall guidance is folded in here so it's not
    repeated inside every mode block."""
    recall_line = (
        "- Use `recall(query=...)` to navigate the codebase before reading files; an index lives at `.squishy/index.json`."
        if has_idx
        else "- No repo index yet. Use targeted `read_file`/`list_directory`/`search_files` to navigate; suggest `/init` to enable `recall`."
    )
    plan_line = (
        "- For non-trivial work call `plan_task` early; after the plan is approved, call `update_plan(step_index=N, status=\"done\")` per step and `finish_plan` once at the end. Don't repeat `update_plan` on the same step."
    )
    return (
        "## Rules\n"
        "- Read files before editing them.\n"
        "- `write_file` is for new files only. Use `edit_file` on anything that already exists.\n"
        "- Use relative paths (working dir is set).\n"
        "- After editing, verify with `run_command` (run tests or the program itself).\n"
        "- Don't re-read a file you've already read unless you need a different range.\n"
        "- `@filename` in user input injects that file inline wrapped in `<file>` tags.\n"
        "- When the task is done, reply with a plain-text summary and no tool call.\n"
        f"{recall_line}\n"
        f"{plan_line}"
    )


def _project_line(project: ProjectInfo) -> str:
    """One-line project summary. Empty fields are skipped."""
    bits = [f"language={project.language}"]
    if project.framework:
        bits.append(f"framework={project.framework}")
    if project.build_command:
        bits.append(f"build=`{project.build_command}`")
    if project.test_command:
        bits.append(f"test=`{project.test_command}`")
    return " · ".join(bits)



def _top_level_files_block(cwd: str) -> str:
    files = _top_level_files(cwd)
    if not files:
        return ""
    return f"\n\n## Top-level files\n{', '.join(files)}\n"
 
 
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


def _mode_block(mode: str, cwd: str) -> str:
    """Per-mode rules.

    Each block is the *delta* on top of `## Rules` — anything already
    in the core ruleset (recall, planning, update_plan, etc.) is not
    repeated. Workflow examples and JSON shape blocks were dropped
    because the tool schemas already document them.
    """
    if mode == "plan":
        return (
            "## Mode: plan (read-only)\n"
            "- For any task that touches files, call `plan_task` first; don't write prose before the plan is approved.\n"
            "- Skip `plan_task` only for trivial reads (e.g. one file, no edits).\n"
            "- Aim for `plan_task` within 2-3 turns: recall → 1-2 targeted reads → plan.\n"
            "- Prefer the dedicated tools (`list_directory`, `read_file`, `search_files`, `glob_files`) — they always work. `run_command` accepts a small read-only allowlist (linters, `git` reads, `pytest --collect-only`, common inspection binaries); the dispatcher lists the exact set if you guess wrong.\n"
            "- The shell already runs in the project root — don't prefix commands with `cd /abs/path && …` (use a relative path or pass `cwd`). `python -c \"…\"` and other arbitrary scripts are rejected; use the dedicated read tools instead.\n"
            "- After approval the user switches you into edits mode to execute the plan."
        )
    if mode == "bench":
        return (
            "## Mode: bench\n"
            "- All tools available. No approval prompts. No `plan_task`/`update_plan`/`finish_plan`.\n"
            "- Workflow: understand → locate → fix → verify → finish.\n"
            "- `save_note` for key findings (bug location, test command, root cause) so they survive compaction.\n"
            "- After editing, run the specific test that exercises the bug. `show_diff` before finishing."
        )
    if mode == "yolo":
        return (
            "## Mode: yolo\n"
            "- All tools available, no approval prompts — be careful with destructive commands.\n"
            "- For non-trivial work follow the plan-then-execute loop from `## Rules` (plan_task → update_plan per step → finish_plan)."
        )
    return (
        "## Mode: edits\n"
        "- `run_command` requires per-call user approval.\n"
        "- If a plan was approved, follow it (see `## Rules` for the update_plan / finish_plan flow)."
    )


def _mcp_block() -> str:
    """Return a system prompt section listing available MCP tools."""
    try:
        from squishy.mcp.tools import get_mcp_tools
        tools = get_mcp_tools()
    except Exception:
        return ""
    if not tools:
        return ""
    lines = [f"- `{t.name}`: {t.description}" for t in tools]
    return (
        "\n## MCP Tools\n"
        "External tools available via MCP (Model Context Protocol):\n"
        + "\n".join(lines) + "\n"
        "Call these tools by name like any built-in tool.\n"
    )


def _index_header(cwd: str) -> str:
    """One-line summary of the cached repo index.

    The previous version emitted a 4-line block (header + stats + ext +
    dirs + age). The same signal fits on one line; recall guidance is
    in `## Rules` so we don't repeat it here.

    Returns an empty string when no `.squishy/index.json` exists.
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
    dir_counts: list[tuple[str, int]] = []
    for node in idx.root.walk():
        if node.kind == "dir" and node.path:
            n = sum(1 for c in node.walk() if c.kind == "file")
            dir_counts.append((node.path, n))
    dir_counts.sort(key=lambda kv: -kv[1])
    top_dirs = ", ".join(f"{p}({n})" for p, n in dir_counts[:5]) or "(flat)"

    age_s = max(0.0, time.time() - (meta.generated_at or 0.0))
    if age_s < 120:
        age = f"{int(age_s)}s"
    elif age_s < 7200:
        age = f"{int(age_s / 60)}m"
    else:
        age = f"{int(age_s / 3600)}h"

    return (
        f"\n\n## Index\n"
        f"{stats.get('files', 0)} files, {stats.get('symbols', 0)} symbols, "
        f"top dirs: {top_dirs} (indexed {age} ago)."
    )
 
 
def _top_level_files(cwd: str, limit: int = 50) -> list[str]:
    if not os.path.isdir(cwd):
        return []
    out = []
    for name in sorted(os.listdir(cwd)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        out.append(name)
        if len(out) >= limit:
            break
    return out
 
 
def snip_old_tool_results(
    messages: list[dict[str, Any]],
    max_chars: int = 2000,
    preserve_last_n: int = 6,
) -> list[dict[str, Any]]:
    """Truncate old tool-role messages that exceed *max_chars*.

    For tool messages older than *preserve_last_n* from the end, keep the
    first half and last quarter of their content, inserting a snip marker.
    Mutates in place and returns the same list.
    """
    cutoff = max(0, len(messages) - preserve_last_n)
    for i in range(cutoff):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        first_half = content[: max_chars // 2]
        last_quarter = content[-(max_chars // 4) :]
        snipped = len(content) - len(first_half) - len(last_quarter)
        m["content"] = f"{first_half}\n[... {snipped} chars snipped ...]\n{last_quarter}"
    return messages


def trim_history(messages: list[dict[str, Any]], max_messages: int = 10) -> list[dict[str, Any]]:
    """Keep system + first user + last (max_messages - 2) messages.

    Ported from atlas-proxy/agent.go:41-50. Preserves initial intent while
    bounding context size.

    Before trimming, applies ``snip_old_tool_results`` to compress large
    tool outputs in older messages — this is free (no LLM call) and
    preserves more useful context per turn.

    Pair-aware: the tail is never allowed to begin with a ``role="tool"``
    message, because an orphan tool result whose matching assistant
    ``tool_calls`` has been sliced off will confuse the LLM (it sees a tool
    result it has no record of requesting, and re-requests the same read).
    Leading tool messages are dropped until we hit an assistant or user turn.

    """
    # Layer 1: snip old tool results before trimming
    snip_old_tool_results(messages)

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(messages) <= max_messages:
        return system + non_system

    if not non_system:
        return system

    first_user_idx = next((i for i, m in enumerate(non_system) if m.get("role") == "user"), 0)
    first_user = [non_system[first_user_idx]]
    remaining_budget = max(1, max_messages - len(system) - len(first_user))
    tail = non_system[-remaining_budget:]
    if tail and tail[0] is first_user[0]:
        tail = tail[1:]

    # Drop leading orphan tool results. Their matching assistant tool_calls
    # message has been trimmed away, so the LLM can't associate the result
    # with a prior action.
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]

    # Semantic anchoring: pull up to 3 anchored messages from the dropped
    # middle section into the retained set.
    if remaining_budget > 0:
        dropped_start = first_user_idx + 1
        dropped_end = len(non_system) - (len(tail) if tail else 0)
        dropped = non_system[dropped_start:dropped_end]
        anchored = [m for m in dropped if m.get("_squishy_anchor")]
        # Only re-inject anchored messages that won't be orphans.
        # A tool message is an orphan if its matching assistant tool_calls
        # message is not in the retained set.
        retained_call_ids: set[str] = set()
        for m in first_user + tail:
            for tc in m.get("tool_calls", []):
                if isinstance(tc, dict):
                    retained_call_ids.add(tc.get("id", ""))
        for m in anchored[:3]:
            if m.get("role") == "tool":
                tcid = m.get("tool_call_id", "")
                if tcid and tcid not in retained_call_ids:
                    continue  # skip orphan tool result
            tail.insert(0, m)

    return system + first_user + tail


# ── Layer 2: LLM-based context compaction ────────────────────────────────


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count from message contents (chars / 4)."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        for tc in m.get("tool_calls", []):
            if isinstance(tc, dict):
                func = tc.get("function", {})
                total += len(func.get("name", "")) + len(func.get("arguments", ""))
    return total // 4


def find_compaction_split(
    messages: list[dict[str, Any]], keep_ratio: float = 0.3
) -> int:
    """Find the index that splits messages so ~keep_ratio of tokens are kept.

    Walks backwards from end, accumulating token estimates, and returns
    the index where the recent portion reaches keep_ratio of total tokens.
    """
    total = _estimate_message_tokens(messages)
    target = int(total * keep_ratio)
    running = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        content = m.get("content", "")
        chars = len(content) if isinstance(content, str) else 0
        for tc in m.get("tool_calls", []):
            if isinstance(tc, dict):
                func = tc.get("function", {})
                chars += len(func.get("name", "")) + len(func.get("arguments", ""))
        running += chars // 4
        if running >= target:
            return i
    return 0


async def compact_messages(
    messages: list[dict[str, Any]],
    client: Any,  # squishy.client.Client
    context_limit: int,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Compress old messages into a summary via LLM call (Layer 2).

    Only fires when estimated token usage exceeds ``context_limit * threshold``.
    Splits messages at a 30% keep ratio, summarizes the old portion, and
    returns ``[system_msgs, summary_msg, ack_msg, *recent]``.

    Anchored messages (``_squishy_anchor``) in the old portion are pulled
    into the recent section to preserve high-value context.
    """
    est = _estimate_message_tokens(messages)
    if est <= int(context_limit * threshold):
        return messages

    # Separate system messages (always preserved as-is)
    def _is_system(m: dict[str, Any]) -> bool:
        return m.get("role") == "system"

    system = [m for m in messages if _is_system(m)]
    non_system = [m for m in messages if not _is_system(m)]

    if len(non_system) < 4:
        return messages

    # Protect the first user message (contains problem statement / task
    # instructions) from being summarized away.
    first_user_idx = next(
        (i for i, m in enumerate(non_system) if m.get("role") == "user"), None,
    )
    if first_user_idx is not None:
        protected = non_system[first_user_idx]
        compactable = non_system[:first_user_idx] + non_system[first_user_idx + 1:]
    else:
        protected = None
        compactable = non_system

    if len(compactable) < 4:
        return messages

    split = find_compaction_split(compactable)
    if split <= 0:
        return messages

    old = compactable[:split]
    recent = compactable[split:]

    # Pull anchored messages from old section into recent
    anchored = [m for m in old if m.get("_squishy_anchor")]
    for m in anchored[:3]:
        recent.insert(0, m)

    # Build summary text from old messages
    summary_parts: list[str] = []
    for m in old:
        if m in anchored:
            continue  # already pulled into recent
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            summary_parts.append(f"[{role}]: {content[:500]}")
        elif m.get("tool_calls"):
            for tc in m["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "?")
                # Include key args (file paths, commands) for context
                args_preview = ""
                try:
                    import json as _json
                    args = _json.loads(func.get("arguments", "{}"))
                    if "path" in args:
                        args_preview = f"path={args['path']}"
                    elif "command" in args:
                        args_preview = f"cmd={str(args['command'])[:80]}"
                    elif "query" in args:
                        args_preview = f"query={args['query']}"
                    elif "pattern" in args:
                        args_preview = f"pattern={args['pattern']}"
                except Exception:  # noqa: BLE001
                    pass
                detail = f"({args_preview})" if args_preview else ""
                summary_parts.append(f"[{role}]: called {name}{detail}")

    old_text = "\n".join(summary_parts)
    # Cap the text sent for summarization to avoid blowing up the compaction prompt.
    if len(old_text) > 30_000:
        old_text = old_text[:30_000] + "\n[... truncated for summarization ...]"

    # Summarize via LLM
    try:
        summary_prompt = (
            "Summarize this conversation history concisely. Preserve: "
            "file paths, function/class names, error messages, test commands, "
            "line numbers, root cause findings, and what was tried. "
            "Be specific about file locations and code details.\n\n"
            + old_text
        )
        result = await client.complete(
            [
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": summary_prompt},
            ],
            tools=[],
            stream=False,
        )
        summary_text = result.text or "(no summary)"
    except Exception:  # noqa: BLE001
        # If summarization fails, fall back to a simple truncation
        summary_text = old_text[:2000] + "\n...(truncated)"

    summary_msg = {
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary_text}",
    }
    ack_msg = {
        "role": "assistant",
        "content": "Understood. I have the context from earlier. Continuing.",
    }

    # Re-inject the protected first user message right after system messages
    # so it survives compaction and remains visible to the model.
    protected_msgs = [protected] if protected is not None else []
    return system + protected_msgs + [summary_msg, ack_msg] + recent
