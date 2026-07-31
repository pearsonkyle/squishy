"""Project detection, system prompt assembly, conversation history trim.
 
Ported from atlas-proxy/project.go and atlas-proxy/agent.go:buildSystemPrompt.
"""
 
from __future__ import annotations
 
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from squishy.index.store import has_index
from squishy.tokens import (
    CHARS_PER_TOKEN,
    PER_MSG_OVERHEAD,
    estimate_message_tokens,
    message_chars,
)
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
    profile: str = "standard",
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
    rules = _rules_block(has_idx, profile)
    mode_block = _mode_block(mode, cwd, profile)
    project_line = _project_line(project)
    index_block = _index_header(cwd)
    top_files_block = "" if has_idx else _top_level_files_block(cwd)
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
    # MCP tools are NOT listed in prose here: they're already in the tool
    # schema (with names + descriptions), so a prose block would only
    # duplicate them and waste context.
    return body + index_block + top_files_block + instructions_block


def _rules_block(has_idx: bool, profile: str = "standard") -> str:
    """Core rules. Recall guidance is folded in here so it's not
    repeated inside every mode block."""
    if profile == "shell":
        return _shell_rules_block()
    if profile == "minimal":
        return _minimal_rules_block(has_idx)
    recall_line = (
        "- Use `recall(query=...)` to navigate the codebase before reading files; an index lives at `.squishy/index.json`."
        if has_idx
        else "- No repo index yet. Use targeted `read_file`/`list_directory`/`search_files` to navigate; suggest `/init` to enable `recall`."
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
        f"{recall_line}"
    )


def _shell_rules_block() -> str:
    """Rules for the `shell` profile — one tool, so almost nothing to say.

    Everything happens through `run_command`, which is the interface these
    models have seen most. The only thing worth stating is how to edit a file
    without an edit tool, since that is the one operation a shell makes
    awkward.
    """
    return (
        "## Rules\n"
        "- `run_command` is your only tool. Use it to read, search, edit, and "
        "run tests.\n"
        "- Inspect code with `cat`, `sed -n '10,40p' file`, `grep -rn`, `ls`.\n"
        "- To change a file, apply a patch or rewrite it — e.g. "
        "`python - <<'EOF'` with a small script, or `cat > file <<'EOF'`. "
        "Verify the change with `git diff` afterwards.\n"
        "- Commands run in the project root; no `cd` prefix needed.\n"
        "- When the task is done, reply with a plain-text summary and no tool call."
    )


def _minimal_rules_block(has_idx: bool) -> str:
    """Rules for the `minimal` tool profile.

    Deliberately short. The profile exposes `run_command`, `read_file`,
    `edit_file`, `write_file` (plus `recall` when an index exists), so there
    is nothing to say about planning, phases, or the browsing tools — the
    shell covers listing, globbing, and grepping. Everything a tool schema
    already documents is omitted rather than restated.
    """
    recall_line = (
        "- `recall(query=...)` searches a prebuilt index of this repo — use it "
        "to locate code before reading files.\n"
        if has_idx else ""
    )
    return (
        "## Rules\n"
        "- Read a file before you edit it.\n"
        "- `edit_file` for existing files, `write_file` only for new ones.\n"
        "- Use relative paths; the shell already runs in the working dir.\n"
        "- `run_command` covers listing, globbing, and grepping — use it for "
        "anything there isn't a dedicated tool for.\n"
        f"{recall_line}"
        "- Verify your change by running the relevant tests.\n"
        "- When the task is done, reply with a plain-text summary and no tool call."
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


def _mode_block(mode: str, cwd: str = "", profile: str = "standard") -> str:
    """Per-mode rules.

    Each block is the *delta* on top of `## Rules` — anything already in the
    core ruleset is not repeated. Workflow examples and JSON shape blocks were
    dropped because the tool schemas already document them.

    The bench block is now the same short task framing for every profile. It
    used to narrate a five-phase state machine to the model; that machine is
    gone, and describing it cost ~250 tokens on every single request.
    """
    if profile in ("minimal", "shell"):
        # Narrow profiles don't expose save_note, so the bench framing below
        # would name a tool they can't call.
        if mode == "bench":
            return (
                "## Task\n"
                "- Change the SOURCE code that implements the behavior. Editing "
                "tests is never a fix.\n"
                "- Don't write reproduction scripts or new test files. Run the "
                "tests named in the task; if one doesn't exist yet, implement "
                "the behavior its name implies rather than hunting for it.\n"
                "- Import and environment errors are not the bug — don't chase them.\n"
                "- Do not stop until you have actually edited a non-test source "
                "file. Ending with no edit is a failed run."
            )
        return ""
    if mode == "bench":
        return (
            "## Task\n"
            "- Change the SOURCE code that implements the behavior. Editing "
            "tests is never a fix.\n"
            "- Don't write reproduction scripts or new test files. Run the "
            "tests named in the task; if one doesn't exist yet, implement the "
            "behavior its name implies rather than hunting for it.\n"
            "- Import and environment errors are not the bug — don't chase them.\n"
            "- Use `save_note` for key findings so they survive context compaction.\n"
            "- Do not stop until you have actually edited a non-test source "
            "file. Ending with no edit is a failed run."
        )
    if mode == "yolo":
        return (
            "## Mode: yolo\n"
            "- All tools available, no approval prompts — be careful with "
            "destructive commands."
        )
    return (
        "## Mode: edits\n"
        "- `run_command` requires per-call user approval."
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

    # Index age intentionally omitted: it's a wall-clock value that would
    # bust vLLM's prefix cache if ``build_system_prompt`` is ever called
    # mid-task (currently it's only called at init, but the dependency was
    # latent). The age provides no actionable signal to the model.
    return (
        f"\n\n## Index\n"
        f"{stats.get('files', 0)} files, {stats.get('symbols', 0)} symbols, "
        f"top dirs: {top_dirs}."
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
    *,
    aggressive_read_file_n: int = 4,
) -> list[dict[str, Any]]:
    """Truncate old tool-role messages that exceed *max_chars*.

    For tool messages older than *preserve_last_n* from the end, keep the
    first half and last quarter of their content, inserting a snip marker.
    Mutates in place and returns the same list.

    For ``read_file`` results older than *aggressive_read_file_n* from the
    end, replace the content entirely with a one-line stub.  read_file
    results dominate context bloat (15KB+ per source file × many files),
    and the agent can always re-read.
    """
    n = len(messages)
    read_cutoff = max(0, n - aggressive_read_file_n)
    cutoff = max(0, n - preserve_last_n)
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue

        # Aggressive read_file stubbing: any read_file result older than
        # aggressive_read_file_n turns and longer than 200 chars gets
        # replaced with a terse marker.  The wording deliberately does
        # NOT invite re-reads (v27 regressed because the prior stub said
        # "call read_file again if needed" and the model treated that as
        # a directive).  Prefer a path stamped on the message at dispatch
        # time over regex-scraping the (possibly truncated) JSON body.
        if (
            i < read_cutoff
            and m.get("name") == "read_file"
            and len(content) > 200
        ):
            path_hint = m.get("_squishy_read_path") or ""
            if not path_hint:
                try:
                    pm = re.search(r'"path"\s*:\s*"([^"]+)"', content[:500])
                    if pm:
                        path_hint = pm.group(1)
                except Exception:  # noqa: BLE001
                    pass
            path_str = f"path={path_hint}" if path_hint else "path=?"
            stub = (
                f"[read_file({path_str}) result elided to save context "
                "— content unchanged on disk]"
            )
            m["content"] = stub
            continue

        if i >= cutoff:
            continue
        if len(content) <= max_chars:
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
        # Even when no trimming happens, strict endpoints (Azure) reject any
        # orphan assistant tool_calls — strip them defensively.
        return system + _strip_orphan_assistant_tool_calls(non_system)

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

    # Strip orphan tool_calls from assistant messages whose matching tool
    # responses were trimmed away. Azure-strict endpoints reject the request
    # otherwise ("tool_call_ids did not have response messages").  Leniently
    # converts the assistant message to prose-only: keeps any text content,
    # drops the dangling tool_calls field.
    tail = _strip_orphan_assistant_tool_calls(tail)

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

    # Re-apply: anchored insertion may have introduced new assistant tool_calls
    # without their paired tool responses (the responses were not anchored).
    tail = _strip_orphan_assistant_tool_calls(tail)

    return system + first_user + tail


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enforce the assistant↔tool pairing invariant on an outgoing message list.

    Two failure modes exist and different sites can introduce either:

    * **Forward orphan** — an ``assistant`` message with ``tool_calls`` whose
      paired ``tool`` responses are missing (trimming/compaction dropped them).
      Handled by :func:`_strip_orphan_assistant_tool_calls`.
    * **Reverse orphan** — a ``role="tool"`` message whose ``tool_call_id`` was
      never declared by any preceding ``assistant`` ``tool_calls`` (a synthetic
      result injected without its paired assistant call, or a tool message left
      at the head of a compaction split). Strict endpoints (Azure/OpenAI) 400 on
      both. This drops reverse orphans.

    Runs once immediately before every ``client.complete`` in the loop, so the
    transcript is well-formed regardless of which nudge/gate mutated it. Returns
    a new list; does not mutate the input.
    """
    msgs = _strip_orphan_assistant_tool_calls(messages)
    declared: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    declared.add(tc["id"])
            out.append(m)
        elif m.get("role") == "tool":
            tcid = m.get("tool_call_id", "")
            if tcid and tcid in declared:
                out.append(m)
            # else: reverse orphan — drop it.
        else:
            out.append(m)
    return _merge_adjacent_same_role(out)


def _merge_adjacent_same_role(
    msgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Coalesce directly-adjacent same-role user/assistant messages.

    Several gates can each append a ``[system]`` nudge (they are sent as
    ``role="user"``) within one turn, producing consecutive user messages.
    Templates that require strict alternation — Mistral/Ministral among them —
    reject the whole request with "conversation roles must alternate user and
    assistant roles", which surfaced live as an APIError mid-run. Merging is
    also a small token win.

    Assistant messages are merged only when neither carries ``tool_calls``, so
    tool-call pairing is never disturbed. Tool messages are left untouched.
    """
    out: list[dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        if role not in ("user", "assistant") or not out:
            out.append(m)
            continue
        prev = out[-1]
        if prev.get("role") != role:
            out.append(m)
            continue
        if role == "assistant" and (prev.get("tool_calls") or m.get("tool_calls")):
            out.append(m)
            continue
        prev_content = prev.get("content")
        cur_content = m.get("content")
        if not isinstance(prev_content, str) or not isinstance(cur_content, str):
            out.append(m)
            continue
        merged = dict(prev)
        joined = "\n\n".join(p for p in (prev_content, cur_content) if p)
        merged["content"] = joined
        out[-1] = merged
    return out


def _strip_orphan_assistant_tool_calls(
    msgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip ``tool_calls`` from assistant messages whose paired tool messages
    are not present later in ``msgs``.

    Azure-strict chat completions reject any assistant message with
    ``tool_calls`` that isn't followed by a ``role="tool"`` message for every
    ``tool_call.id``.  When trimming/anchoring drops some tool responses while
    keeping their assistant message, the result is an orphan.

    This converts the orphan to a prose-only assistant message (preserving
    ``content`` and ``think``).  If the message has neither content nor
    reasoning, drops the message entirely so we don't leave an empty turn.
    """
    out: list[dict[str, Any]] = []
    n = len(msgs)
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            out.append(m)
            continue
        tcs = m.get("tool_calls") or []
        if not tcs:
            out.append(m)
            continue
        # Collect tool_call_ids that appear as tool messages later in tail.
        following_tool_ids: set[str] = set()
        for j in range(i + 1, n):
            mj = msgs[j]
            if mj.get("role") == "tool":
                tcid = mj.get("tool_call_id", "")
                if tcid:
                    following_tool_ids.add(tcid)
            elif mj.get("role") == "assistant":
                # Stop scan at the next assistant turn — tool responses for
                # *this* assistant message must come before any other assistant.
                break
        required_ids = {
            tc.get("id", "")
            for tc in tcs
            if isinstance(tc, dict) and tc.get("id")
        }
        if required_ids.issubset(following_tool_ids):
            out.append(m)
            continue
        # Orphan: at least one tool_call.id has no following tool message.
        # Convert to prose-only assistant message.
        text = m.get("content") or ""
        think = m.get("think") or ""
        if not text and not think:
            # Nothing salvageable — drop the assistant turn entirely.
            continue
        cleaned = {k: v for k, v in m.items() if k != "tool_calls"}
        cleaned["content"] = text or ""
        out.append(cleaned)
    return out


# ── Layer 2: LLM-based context compaction ────────────────────────────────


# Token estimation lives in squishy.tokens (single source of the 3.5
# chars/token heuristic). Kept as a module-local alias for readability and
# back-compat with any importer of _estimate_message_tokens.
_estimate_message_tokens = estimate_message_tokens


def find_compaction_split(
    messages: list[dict[str, Any]], keep_ratio: float = 0.3
) -> int:
    """Find the index that splits messages so ~keep_ratio of tokens are kept.

    Walks backwards from end, accumulating token estimates, and returns
    the index where the recent portion reaches keep_ratio of total tokens.
    """
    total = estimate_message_tokens(messages)
    target = int(total * keep_ratio)
    running = 0
    for i in range(len(messages) - 1, -1, -1):
        running += int(message_chars(messages[i]) / CHARS_PER_TOKEN) + PER_MSG_OVERHEAD
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

    # Pull anchored messages from old section into recent, together with
    # their paired tool-result messages so we don't create dangling
    # tool_calls that confuse the LLM.
    anchored = [m for m in old if m.get("_squishy_anchor")]
    pulled: list[dict[str, Any]] = []
    for m in anchored[:3]:
        pulled.append(m)
        # If this is an assistant message with tool_calls, also pull the
        # matching tool-result messages from old.
        if m.get("role") == "assistant" and m.get("tool_calls"):
            call_ids = {
                tc.get("id", "") for tc in m["tool_calls"]
                if isinstance(tc, dict)
            }
            for om in old:
                if (
                    om.get("role") == "tool"
                    and om.get("tool_call_id", "") in call_ids
                    and om not in pulled
                ):
                    pulled.append(om)
    for m in reversed(pulled):
        recent.insert(0, m)

    # Build summary text from old messages
    pulled_set = set(id(m) for m in pulled)
    summary_parts: list[str] = []
    for m in old:
        if id(m) in pulled_set:
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
                    args = json.loads(func.get("arguments", "{}"))
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

    # Summarize via LLM.  Use a structured template so the model produces
    # decision-relevant facts instead of narrative prose.  Same token
    # budget, ~3-4× more useful signal per token.
    try:
        summary_prompt = (
            "Compress the following conversation history into the structured "
            "template below.  Be concrete: include file paths, line numbers, "
            "exact error messages, and the literal text of attempted edits.\n"
            "Do NOT include narrative prose.  If a section has no content, "
            "write `(none)`.  Stay under 600 words total.\n\n"
            "ROOT CAUSE (1-2 sentences, what the bug actually is):\n"
            "ATTEMPTED EDITS (file:line — what changed — succeeded/failed):\n"
            "FILES READ (path — one-line relevance, no prose):\n"
            "TESTS RUN (command — exit_code — pass/fail summary):\n"
            "DEAD ENDS (approaches tried and ruled out, with reason):\n"
            "NEXT ACTION (the single thing to do next):\n\n"
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
    # The split is chosen purely by token count, so `recent` can begin with a
    # `role="tool"` message whose assistant tool_calls was summarized into
    # `old` — a reverse orphan the strict endpoints reject. Drop any such
    # leading tool messages (mirrors trim_history's tail guard).
    while recent and recent[0].get("role") == "tool":
        recent = recent[1:]
    # Final orphan-strip on `recent`: the anchored-pull above tries to keep
    # tool-result pairs together, but if any tool messages were dropped
    # mid-conversation an orphan can remain.  Strict endpoints (Azure) reject.
    recent = _strip_orphan_assistant_tool_calls(recent)
    return system + protected_msgs + [summary_msg, ack_msg] + recent
