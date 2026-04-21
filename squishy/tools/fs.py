"""Filesystem tools (async). FS IO is fast; we keep it synchronous inside
async functions rather than thread-dispatching every call.
 
For search_files we shell out to ripgrep via asyncio.create_subprocess_exec
so long searches don't block the loop.
"""
 
from __future__ import annotations
 
import asyncio
import fnmatch
import os
import re
import shutil
from typing import Any
 
from squishy.tools.base import Tool, ToolContext, ToolResult
 
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
WRITE_LIMIT_LINES = 100
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
 
 
async def _read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    if not isinstance(path, str):
        return ToolResult(False, error="`path` is required (string)")
    abs_path = _resolve(path, ctx.working_dir)
    if not os.path.isfile(abs_path):
        return ToolResult(False, error=f"file not found: {path}")

    offset = int(args.get("offset") or 0)
    limit = args.get("limit")

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
    return ToolResult(
        True,
        data={
            "path": path,
            "content": content,
            "total_lines": len(lines),
            "returned_lines": len(sliced),
            "offset": offset,
        },
        display=f"{len(sliced)} lines loaded",
    )
 
 
async def _write_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    content = args.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return ToolResult(False, error="`path` and `content` are required strings")
    abs_path = _resolve(path, ctx.working_dir)
 
    if os.path.isfile(abs_path):
        with open(abs_path, encoding="utf-8", errors="replace") as _f:
            existing_lines = sum(1 for _ in _f)
        if existing_lines > WRITE_LIMIT_LINES:
            return ToolResult(
                False,
                error=(
                    f"refusing write_file: {path} exists with {existing_lines} lines (>{WRITE_LIMIT_LINES}). "
                    "Use edit_file with old_str/new_str for targeted changes."
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
        return ToolResult(False, error="old_str not found in file")
    if count > 1 and not replace_all:
        return ToolResult(
            False,
            error=(
                f"old_str matches {count} times; pass replace_all=true or expand old_str "
                "with more surrounding context to make it unique."
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
 
    rg = shutil.which("rg")
    if rg:
        return await _rg_search(rg, pattern, abs_path, glob)
    return await asyncio.to_thread(_python_search, pattern, abs_path, glob)
 
 
async def _rg_search(
    rg: str, pattern: str, abs_path: str, glob: Any
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
    for line in stdout.decode("utf-8", errors="replace").splitlines()[:SEARCH_CAP]:
        parts = line.split(":", 2)
        if len(parts) == 3:
            matches.append({"file": parts[0], "line": int(parts[1]), "text": parts[2]})
    return ToolResult(
        True,
        data={"pattern": pattern, "matches": matches, "count": len(matches)},
        display=f"{len(matches)} matches",
    )
 
 
def _python_search(pattern: str, abs_path: str, glob: Any) -> ToolResult:
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
                            if len(matches) >= SEARCH_CAP:
                                break
            except OSError:
                continue
            if len(matches) >= SEARCH_CAP:
                break
        if len(matches) >= SEARCH_CAP:
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
    mutates=False,
)
 
write_file = Tool(
    name="write_file",
    description="Write an entire file. Rejects existing files longer than "
                f"{WRITE_LIMIT_LINES} lines — use edit_file instead.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    },
    run=_write_file,
    mutates=True,
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
    mutates=True,
)
 
list_directory = Tool(
    name="list_directory",
    description="List files and directories. Hides dotfiles and standard vendor dirs.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "default": "."}},
    },
    run=_list_directory,
    mutates=False,
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
    mutates=False,
)
 
FS_TOOLS: list[Tool] = [read_file, write_file, edit_file, list_directory, search_files]