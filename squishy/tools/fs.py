"""Filesystem tools (async). FS IO is fast; we keep it synchronous inside
async functions rather than thread-dispatching every call.
 
For search_files we shell out to ripgrep via asyncio.create_subprocess_exec
so long searches don't block the loop.
"""
 
from __future__ import annotations
 
import asyncio
import difflib
import fnmatch
import os
import re
import shutil
from typing import Any
 
from squishy.tools.base import Tool, ToolContext, ToolResult
 
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
SEARCH_TIMEOUT = 15.0
SEARCH_CAP = 200
 
 
def _resolve(path: str, cwd: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(cwd, path))


def _invalidate_read_cache(ctx: ToolContext, path: str) -> None:
    """Drop any cached reads for `path` after a mutating write/edit."""
    for key in [k for k in ctx.files_read_meta if k[0] == path]:
        del ctx.files_read_meta[key]
    ctx.files_read.pop(path, None)


def _collect_match_context(
    text: str, needle: str, *, max_matches: int = 3, context_lines: int = 2
) -> str:
    """Return a short string showing up to ``max_matches`` match sites with
    ``context_lines`` lines of surrounding context. Used by edit_file to help
    the model disambiguate without re-reading the entire file."""
    lines = text.splitlines()
    needle_first_line = needle.splitlines()[0] if needle else needle
    out: list[str] = []
    found = 0
    for i, line in enumerate(lines):
        if needle_first_line not in line:
            continue
        start = max(0, i - context_lines)
        end = min(len(lines), i + context_lines + 1)
        chunk = [f"  L{n + 1}: {lines[n]}" for n in range(start, end)]
        out.append(f"--- match {found + 1} at line {i + 1} ---\n" + "\n".join(chunk))
        found += 1
        if found >= max_matches:
            break
    return "\n".join(out) if out else "(no match context available)"
 
 
async def _read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    if not isinstance(path, str):
        return ToolResult(False, error="`path` is required (string)")
    abs_path = _resolve(path, ctx.working_dir)
    if not os.path.isfile(abs_path):
        return ToolResult(False, error=f"file not found: {path}")

    offset = int(args.get("offset") or 0)
    limit = args.get("limit")

    # Hard cap: refuse after too many reads of the same path.
    path_count = ctx.files_read_count.get(path, 0)
    if path_count >= 5:
        return ToolResult(
            False,
            error=(
                f"Refused: you have already read '{path}' {path_count} times. "
                "You have the content — use `save_note` to persist key parts if needed, "
                "then call `edit_file` with your fix. Do NOT read this file again."
            ),
        )

    # Duplicate-read dedup. When the same path is requested with the same
    # offset+limit window as a prior read in this conversation, return the
    # cached content with an unmistakable marker so the LLM realizes it has
    # already seen this file. This prevents the model from re-requesting
    # reads after earlier turns were trimmed from history.
    cache_key = (path, offset, limit)
    prior = ctx.files_read_meta.get(cache_key)
    if prior is not None:
        return ToolResult(
            True,
            data={
                "path": path,
                "content": prior["content"],
                "total_lines": prior["total_lines"],
                "returned_lines": prior["returned_lines"],
                "offset": offset,
                "cache_hit": True,
                "note": (
                    "You already read this file earlier in this conversation with the same "
                    "offset/limit. Use what you have. Only re-call read_file with a different "
                    "offset/limit if you need a different range."
                ),
            },
            display=f"cache hit ({prior['returned_lines']} lines, already read)",
        )

    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return ToolResult(False, error=str(e))

    lines = text.splitlines()
    sliced = lines[offset : offset + int(limit)] if limit is not None else lines[offset:]
    content = "\n".join(sliced)

    ctx.files_read[path] = content
    ctx.files_read_meta[cache_key] = {
        "content": content,
        "total_lines": len(lines),
        "returned_lines": len(sliced),
    }
    # Track total reads per path (regardless of offset/limit).
    ctx.files_read_count[path] = ctx.files_read_count.get(path, 0) + 1
    path_reads = ctx.files_read_count[path]

    data: dict[str, Any] = {
        "path": path,
        "content": content,
        "total_lines": len(lines),
        "returned_lines": len(sliced),
        "offset": offset,
    }
    if path_reads >= 3:
        data["warning"] = (
            f"This is read #{path_reads} of '{path}'. You are reading this file "
            "too many times. Use `save_note` to persist key content, or use "
            "`recall` to find a different file if this isn't the right one."
        )
    return ToolResult(
        True,
        data=data,
        display=f"{len(sliced)} lines loaded",
    )
 
 
async def _write_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    content = args.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return ToolResult(False, error="`path` and `content` are required strings")
    abs_path = _resolve(path, ctx.working_dir)

    # Hard guard: write_file is for creating NEW files only. Existing files
    # must be modified with edit_file — full rewrites via write_file are the
    # main tool-misuse pathology observed in small-model coding sessions.
    if os.path.isfile(abs_path):
        return ToolResult(
            False,
            error=(
                f"write_file refused — {path} already exists.\n"
                "\n"
                "write_file is only for creating NEW files. To change an existing file, use edit_file:\n"
                f'  edit_file(path="{path}", old_str="<exact text>", new_str="<replacement>")\n'
                "\n"
                "If you don't know the current content, read_file it first. Include 2-3 surrounding\n"
                "lines to make old_str unique. For multiple changes, call edit_file multiple times.\n"
                "Do NOT retry write_file."
            ),
        )

    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    _invalidate_read_cache(ctx, path)
    return ToolResult(
        True,
        data={"path": path, "bytes": len(content.encode("utf-8"))},
        display=f"wrote {len(content.encode('utf-8'))} bytes",
    )
 
 
async def _edit_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    old_str = args.get("old_str")
    new_str = args.get("new_str")
    replace_all = bool(args.get("replace_all", False))
    if not all(isinstance(x, str) for x in (path, old_str, new_str)):
        return ToolResult(False, error="`path`, `old_str`, `new_str` are required strings")
    assert isinstance(path, str) and isinstance(old_str, str) and isinstance(new_str, str)
 
    abs_path = _resolve(path, ctx.working_dir)
    if not os.path.isfile(abs_path):
        return ToolResult(False, error=f"file not found: {path}")
 
    with open(abs_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
 
    count = text.count(old_str)
    if count == 0:
        # Stage 1: try stripping trailing whitespace from each line
        text_stripped = "\n".join(line.rstrip() for line in text.split("\n"))
        old_stripped = "\n".join(line.rstrip() for line in old_str.split("\n"))
        stripped_count = text_stripped.count(old_stripped)

        if stripped_count == 1:
            new_stripped = "\n".join(line.rstrip() for line in new_str.split("\n"))
            new_text = text_stripped.replace(old_stripped, new_stripped, 1)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            _invalidate_read_cache(ctx, path)
            old_lines = len(old_str.splitlines()) or 1
            new_lines = len(new_str.splitlines()) or 1
            return ToolResult(
                True,
                data={
                    "path": path,
                    "replacements": 1,
                    "old_lines": old_lines,
                    "new_lines": new_lines,
                    "old_str": old_str,
                    "new_str": new_str,
                    "note": "trailing whitespace normalized",
                },
                display=f"{old_lines} -> {new_lines} lines (trailing whitespace normalized)",
            )

        if stripped_count > 1:
            context_snippets = _collect_match_context(
                text_stripped, old_stripped, max_matches=3, context_lines=2
            )
            return ToolResult(
                False,
                error=(
                    f"old_str not found exactly, but after normalizing trailing whitespace "
                    f"it matches {stripped_count} times. Add more surrounding context to make "
                    f"it unique, or use replace_all=true. Match sites:\n"
                    + context_snippets
                ),
            )

        # Stage 2: no match even after normalization — provide diagnostic hint
        old_lines_list = old_str.split("\n")
        file_lines = text.split("\n")
        hint = ""

        # Try to find the first line of old_str and show actual content at that location
        if old_lines_list and old_lines_list[0].strip():
            needle = old_lines_list[0].strip()
            for i, fl in enumerate(file_lines):
                if needle in fl.strip():
                    # Show the actual content at this location for the same number of lines
                    num_old_lines = len(old_lines_list)
                    actual_lines = file_lines[i : i + num_old_lines]
                    actual_block = "\n".join(actual_lines)
                    hint = (
                        f" The first line of old_str appears at line {i + 1}, but the full "
                        f"block differs. Actual content at lines {i + 1}-{i + len(actual_lines)}:\n"
                        f"---\n{actual_block}\n---\n"
                        f"Re-call edit_file with the exact text above as old_str."
                    )
                    break

        # Stage 3: fuzzy match — find a contiguous block that closely matches old_str
        if not hint and len(old_lines_list) >= 2:
            best_ratio = 0.0
            best_start = -1
            num_old = len(old_lines_list)
            for i in range(len(file_lines) - num_old + 1):
                candidate = file_lines[i : i + num_old]
                ratio = difflib.SequenceMatcher(
                    None, old_str, "\n".join(candidate)
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
            if best_ratio >= 0.85 and best_start >= 0:
                actual_lines = file_lines[best_start : best_start + num_old]
                actual_block = "\n".join(actual_lines)
                hint = (
                    f" A similar block was found at lines {best_start + 1}-"
                    f"{best_start + num_old} ({best_ratio:.0%} match):\n"
                    f"---\n{actual_block}\n---\n"
                    f"Re-call edit_file with the exact text above as old_str."
                )

        if not hint:
            hint = " Read the file first and copy the exact text."

        return ToolResult(
            False,
            error=f"old_str not found in file.{hint}",
        )
    if count > 1 and not replace_all:
        context_snippets = _collect_match_context(text, old_str, max_matches=3, context_lines=2)
        return ToolResult(
            False,
            error=(
                f"old_str matches {count} times; pass replace_all=true or expand old_str "
                "with more surrounding context to make it unique. Match sites:\n"
                + context_snippets
            ),
        )
 
    new_text = text.replace(old_str, new_str, -1 if replace_all else 1)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    _invalidate_read_cache(ctx, path)
    old_lines = len(old_str.splitlines()) or 1
    new_lines = len(new_str.splitlines()) or 1
    return ToolResult(
        True,
        data={
            "path": path,
            "replacements": count if replace_all else 1,
            "old_lines": old_lines,
            "new_lines": new_lines,
            "old_str": old_str,
            "new_str": new_str,
        },
        display=f"{old_lines} → {new_lines} lines",
    )
 
 
async def _list_directory(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path", ".")
    if not isinstance(path, str):
        return ToolResult(False, error="`path` must be a string")
    abs_path = _resolve(path, ctx.working_dir)
    if not os.path.isdir(abs_path):
        return ToolResult(False, error=f"not a directory: {path}")
 
    entries = []
    for name in sorted(os.listdir(abs_path)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        full = os.path.join(abs_path, name)
        kind = "dir" if os.path.isdir(full) else "file"
        size = os.path.getsize(full) if kind == "file" else 0
        entries.append({"name": name, "type": kind, "size": size})
    return ToolResult(
        True,
        data={"path": path, "entries": entries},
        display=f"{len(entries)} entries",
    )
 
 
async def _search_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str):
        return ToolResult(False, error="`pattern` is required (string regex)")
    search_path = args.get("path", ".")
    glob = args.get("glob")
 
    abs_path = (
        _resolve(search_path, ctx.working_dir) if isinstance(search_path, str) else ctx.working_dir
    )
    if not os.path.exists(abs_path):
        return ToolResult(False, error=f"path not found: {search_path}")
 
    cap = 50 if ctx.permission_mode == "bench" else SEARCH_CAP
    rg = shutil.which("rg")
    if rg:
        return await _rg_search(rg, pattern, abs_path, glob, cap=cap)
    return await asyncio.to_thread(_python_search, pattern, abs_path, glob, cap=cap)
 
 
async def _rg_search(
    rg: str, pattern: str, abs_path: str, glob: Any, *, cap: int = SEARCH_CAP,
) -> ToolResult:
    cmd = [rg, "-n", "--no-heading", "-S", pattern, abs_path]
    if isinstance(glob, str):
        cmd.extend(["-g", glob])
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=SEARCH_TIMEOUT)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(False, error="ripgrep timed out")
    except FileNotFoundError as e:
        return ToolResult(False, error=str(e))
 
    matches: list[dict[str, Any]] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines()[:cap]:
        parts = line.split(":", 2)
        if len(parts) == 3:
            matches.append({"file": parts[0], "line": int(parts[1]), "text": parts[2]})
    return ToolResult(
        True,
        data={"pattern": pattern, "matches": matches, "count": len(matches)},
        display=f"{len(matches)} matches",
    )
 
 
def _python_search(pattern: str, abs_path: str, glob: Any, *, cap: int = SEARCH_CAP) -> ToolResult:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(False, error=f"invalid regex: {e}")
    matches: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(abs_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if isinstance(glob, str) and not fnmatch.fnmatch(name, glob):
                continue
            full = os.path.join(root, name)
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            matches.append({"file": full, "line": i, "text": line.rstrip()})
                            if len(matches) >= cap:
                                break
            except OSError:
                continue
            if len(matches) >= cap:
                break
        if len(matches) >= cap:
            break
 
    return ToolResult(
        True,
        data={"pattern": pattern, "matches": matches, "count": len(matches)},
        display=f"{len(matches)} matches",
    )
 
 
read_file = Tool(
    name="read_file",
    description="Read a file from disk. Returns its content and line count. "
                "Use offset/limit to page through large files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute path"},
            "offset": {"type": "integer", "description": "Line offset (0-based)", "default": 0},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    },
    run=_read_file,
)
 
write_file = Tool(
    name="write_file",
    description="Create a new file. Refuses if the file already exists — use edit_file for existing files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    },
    run=_write_file,
)
 
edit_file = Tool(
    name="edit_file",
    description="Replace a unique substring in a file. Use for targeted changes in existing files. "
                "Set replace_all=true to replace every occurrence.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_str": {"type": "string", "description": "Exact text to find (must be unique unless replace_all=true)"},
            "new_str": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_str", "new_str"],
    },
    run=_edit_file,
)
 
list_directory = Tool(
    name="list_directory",
    description="List files and directories. Hides dotfiles and standard vendor dirs.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "default": "."}},
    },
    run=_list_directory,
)
 
search_files = Tool(
    name="search_files",
    description="Grep-style regex search. Uses ripgrep when available.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex"},
            "path": {"type": "string", "default": "."},
            "glob": {"type": "string", "description": "Optional filename glob (e.g. '*.py')"},
        },
        "required": ["pattern"],
    },
    run=_search_files,
)
 
DIFF_CAP = 4000


async def _show_diff(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Show git diff for a file or the entire workspace."""
    path = args.get("path")
    cmd = ["git", "diff"]
    if isinstance(path, str) and path.strip():
        cmd += ["--", _resolve(path, ctx.working_dir)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=ctx.working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except FileNotFoundError:
        return ToolResult(False, error="git is not available")
    except TimeoutError:
        return ToolResult(False, error="git diff timed out")

    if proc.returncode != 0:
        err = stderr_b.decode("utf-8", errors="replace").strip()
        return ToolResult(False, error=f"git diff failed: {err}")

    diff_text = stdout_b.decode("utf-8", errors="replace")
    if not diff_text.strip():
        return ToolResult(True, data={"diff": "", "path": path or "."}, display="no changes")

    truncated = False
    if len(diff_text) > DIFF_CAP:
        diff_text = diff_text[:DIFF_CAP] + "\n…(truncated)"
        truncated = True

    return ToolResult(
        True,
        data={"diff": diff_text, "path": path or ".", "truncated": truncated},
        display=f"{len(diff_text)} chars" + (" (truncated)" if truncated else ""),
    )


show_diff = Tool(
    name="show_diff",
    description=(
        "Show git diff of your changes. Call after editing files to verify "
        "your changes look correct. Pass a specific path or omit for all changes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to diff (omit for all changes)",
            },
        },
    },
    run=_show_diff,
)


GLOB_CAP = 200


async def _glob_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Find files matching a glob pattern recursively."""
    pattern = args.get("pattern")
    if not isinstance(pattern, str):
        return ToolResult(False, error="`pattern` is required (string glob)")
    path = args.get("path", ".")
    abs_path = _resolve(path if isinstance(path, str) else ".", ctx.working_dir)
    if not os.path.isdir(abs_path):
        return ToolResult(False, error=f"not a directory: {path}")

    from pathlib import Path

    base = Path(abs_path)
    matches: list[str] = []
    try:
        for p in sorted(base.glob(pattern)):
            # Skip hidden files and vendor directories
            parts = p.relative_to(base).parts
            if any(part.startswith(".") or part in SKIP_DIRS for part in parts):
                continue
            if p.is_file():
                matches.append(str(p.relative_to(base)))
            if len(matches) >= GLOB_CAP:
                break
    except (ValueError, OSError) as e:
        return ToolResult(False, error=f"glob error: {e}")

    return ToolResult(
        True,
        data={"pattern": pattern, "path": path, "matches": matches, "count": len(matches)},
        display=f"{len(matches)} files",
    )


glob_files = Tool(
    name="glob_files",
    description=(
        "Find files matching a glob pattern recursively. "
        "Use ** for recursive matching (e.g., '**/test_*.py', 'src/**/*.js')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py', 'tests/**/test_*.py')",
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "Base directory to search from",
            },
        },
        "required": ["pattern"],
    },
    run=_glob_files,
)


FS_TOOLS: list[Tool] = [read_file, write_file, edit_file, list_directory, search_files, show_diff, glob_files]