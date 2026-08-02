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
import tempfile
from typing import Any
 
from squishy.tool_restrictions import profile_shows
from squishy.tools.base import Tool, ToolContext, ToolResult
 
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}
SEARCH_TIMEOUT = 15.0
SEARCH_CAP = 200
 
 
def _resolve(path: str, cwd: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(cwd, path))


def _safe_resolve(path: str, cwd: str) -> tuple[str, str | None]:
    """Resolve *path* relative to *cwd* and verify it stays inside *cwd*.

    Returns ``(abs_path, None)`` on success, or ``("", error_msg)`` when the
    resolved path escapes the working directory.
    """
    abs_path = _resolve(path, cwd)
    real_abs = os.path.realpath(abs_path)
    real_cwd = os.path.realpath(cwd)
    try:
        os.path.commonpath([real_abs, real_cwd])
    except ValueError:
        return "", _escape_error(path, cwd)
    if not (real_abs == real_cwd or real_abs.startswith(real_cwd + os.sep)):
        return "", _escape_error(path, cwd)
    return abs_path, None


# Both the system temp directory and a literal /tmp. They are the same
# directory on Linux (so, in every bench container), but not on macOS, where
# `gettempdir()` is a per-user path under /var/folders and /tmp resolves to
# /private/tmp. Every prompt in this repo says "/tmp" — honoring only
# `gettempdir()` would make the instruction work in the container and fail on
# the machine the developer is testing on.
_SCRATCH_DIRS = tuple({
    os.path.realpath(p)
    for p in (tempfile.gettempdir(), "/tmp")
    if os.path.isdir(p)
})


def _is_scratch(path: str) -> bool:
    """True for an absolute path under a scratch directory.

    The bench prompt and `write_file`'s own refusal both tell the model to put
    reproduction scripts in /tmp, so they stay out of the graded diff. Until
    this existed, following that instruction returned "path outside working
    directory" — the harness refusing the exact action it had just demanded,
    which is the failure mode this codebase keeps rediscovering.
    """
    if not os.path.isabs(path):
        return False
    real = os.path.realpath(os.path.normpath(path))
    return any(
        real == root or real.startswith(root + os.sep) for root in _SCRATCH_DIRS
    )


def _resolve_writable(path: str, cwd: str) -> tuple[str, str | None, bool]:
    """``(abs_path, error, is_scratch)`` — repo paths plus the scratch dir."""
    if _is_scratch(path):
        return os.path.normpath(path), None, True
    abs_path, err = _safe_resolve(path, cwd)
    return abs_path, err, False


# Directories never worth walking when hunting for a mistyped path.
_MISS_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".squishy", "node_modules", "__pycache__",
    "venv", ".venv", "target", "build", "dist", ".tox", ".mypy_cache",
    ".pytest_cache", "vendor", ".idea", ".gradle",
})
_MISS_MAX_CANDIDATES = 5


def _path_candidates(path: str, cwd: str) -> list[str]:
    """Plausible cwd-relative paths the model *meant* by *path*.

    A bare "file not found" is a dead end: observed live, a model that wanted
    `vyper/ast/natspec.py` asked for `vyper/vyper/ast/natspec.py` (the repo
    directory and the package share a name) and burned ~20 turns never
    recovering, because nothing in the error told it what was wrong.

    Two cheap, high-yield repairs, then a bounded basename search:
      * strip a leading component that duplicates the repo directory name;
      * reinterpret a rooted path as repo-relative (`/vyper/x` -> `x`).
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(rel: str) -> None:
        rel = rel.strip("/")
        if not rel or rel in seen:
            return
        if os.path.isfile(os.path.join(cwd, rel)):
            seen.add(rel)
            out.append(rel)

    root = os.path.basename(os.path.realpath(cwd))
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]

    # `<root>/rest` and `/<root>/rest` -> `rest`
    if parts and parts[0] == root:
        add("/".join(parts[1:]))
    # A rooted path that is really repo-relative.
    if os.path.isabs(path):
        add("/".join(parts))
        for i in range(1, len(parts)):
            add("/".join(parts[i:]))
    # Any doubled component (`a/a/b` -> `a/b`).
    for i in range(len(parts) - 1):
        if parts[i] == parts[i + 1]:
            add("/".join(parts[:i] + parts[i + 1:]))

    if out:
        return out[:_MISS_MAX_CANDIDATES]

    # Fall back to finding the basename anywhere in the tree.
    target = parts[-1] if parts else ""
    if not target:
        return []
    for dirpath, dirnames, filenames in os.walk(cwd):
        dirnames[:] = [d for d in dirnames
                       if d not in _MISS_SKIP_DIRS and not d.startswith(".")]
        if target in filenames:
            add(os.path.relpath(os.path.join(dirpath, target), cwd))
            if len(out) >= _MISS_MAX_CANDIDATES:
                break
    return out[:_MISS_MAX_CANDIDATES]


def _escape_error(path: str, cwd: str) -> str:
    """An 'outside the working directory' message with a way forward.

    Absolute paths the model invents are usually the right file under a wrong
    root (`/test/fast-check/src/x.ts` for a repo checked out elsewhere). The
    refusal must stand — but pointing at the real path costs nothing and saves
    the model from guessing another root.
    """
    base = f"path outside working directory: {path}"
    candidates = _path_candidates(path, cwd)
    if candidates:
        return f"{base}. Did you mean: {', '.join(candidates)}?"
    return f"{base}. Paths must be relative to {cwd}."


def _not_found_error(path: str, cwd: str, abs_path: str) -> str:
    """A 'file not found' the model can actually act on."""
    if os.path.isdir(abs_path):
        try:
            entries = sorted(os.listdir(abs_path))[:20]
        except OSError:
            entries = []
        listing = f" Contains: {', '.join(entries)}" if entries else ""
        return (f"{path} is a directory, not a file. Pass a file path, or use "
                f"`list_directory` to browse it.{listing}")
    candidates = _path_candidates(path, cwd)
    if candidates:
        return (f"file not found: {path}. Did you mean: "
                f"{', '.join(candidates)}?")
    # `foo.py` that is really the package `foo/`. Common where a module was
    # split into a package: observed live, a model asked for
    # `src/wtforms/widgets.py` four times across two arms, each time getting a
    # bare "file not found" while `src/wtforms/widgets/` sat right there. The
    # module list is the actual answer, so return it rather than the generic
    # "paths are relative to..." boilerplate.
    pkg = _package_for_module(path, cwd)
    if pkg is not None:
        pkg_path, modules = pkg
        listing = f" It contains: {', '.join(modules)}." if modules else ""
        return (f"file not found: {path}, but `{pkg_path}` is a package "
                f"directory — that module was split into a package.{listing}")
    return (f"file not found: {path}. Paths are relative to the working "
            f"directory ({cwd}) — don't prefix them with the repo name.")


def _package_for_module(path: str, cwd: str) -> tuple[str, list[str]] | None:
    """If ``path`` looks like ``foo.py`` and ``foo/`` is a directory, return
    ``(foo/, [its module files])``. Otherwise None.

    Applies the same leading-component repairs as :func:`_path_candidates`, so
    ``/wtforms/src/wtforms/widgets.py`` resolves against ``src/wtforms/widgets/``
    even though the model rooted it at the repo name.
    """
    base, ext = os.path.splitext(path)
    if not ext or not base:
        return None

    root = os.path.basename(os.path.realpath(cwd))
    parts = [p for p in base.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts:
        return None

    rels: list[str] = ["/".join(parts)]
    if parts[0] == root:
        rels.append("/".join(parts[1:]))
    if os.path.isabs(base):
        rels.extend("/".join(parts[i:]) for i in range(1, len(parts)))

    for rel in rels:
        rel = rel.strip("/")
        if not rel:
            continue
        cand = os.path.join(cwd, rel)
        if os.path.isdir(cand):
            try:
                names = sorted(
                    n for n in os.listdir(cand)
                    if n.endswith(ext) and not n.startswith(".")
                )[:20]
            except OSError:
                names = []
            return rel, names
    return None


def _not_a_dir_error(path: str, cwd: str, abs_path: str) -> str:
    """A 'not a directory' the model can act on.

    Two real cases from the sweep: the model passed a *file* where a directory
    was wanted, and it dropped an extension (`src/printer` for
    `src/printer.ts`). A bare "not a directory" leaves it guessing.
    """
    if os.path.isfile(abs_path):
        return (f"{path} is a file, not a directory. Use `read_file` to read "
                f"it, or pass its parent directory.")
    candidates = _path_candidates(path, cwd)
    if candidates:
        return (f"not a directory: {path}. These files exist though: "
                f"{', '.join(candidates)} — use `read_file` for those.")
    parent = os.path.dirname(path.rstrip("/"))
    parent_abs = os.path.join(cwd, parent) if parent else cwd
    if os.path.isdir(parent_abs):
        try:
            near = sorted(os.listdir(parent_abs))[:20]
        except OSError:
            near = []
        if near:
            return (f"not a directory: {path}. {parent or '.'} contains: "
                    f"{', '.join(near)}")
    return f"not a directory: {path}. Paths are relative to {cwd}."


def _unescape_str(s: str) -> str:
    """Unescape over-escaped characters in model-generated strings.

    Models sometimes double-escape JSON strings, producing literal backslash
    sequences like ``\\"`` or ``\\n`` that don't match actual file content.
    This converts common escape sequences to their actual characters.
    Only applied when the original string doesn't match; safe because real
    backslash sequences in source code would already have matched.
    """
    if "\\" not in s:
        return s
    result = s
    result = result.replace('\\"', '"')
    result = result.replace("\\'", "'")
    result = result.replace("\\n", "\n")
    result = result.replace("\\t", "\t")
    # Don't unescape \\ → \ here (that would break actual backslash content);
    # see _collapse_double_backslash for the last-resort handler.
    return result


def _collapse_double_backslash(s: str) -> str:
    """Collapse literal ``\\\\`` → ``\\`` in model-generated strings.

    Used only as a last-resort fallback after _unescape_str fails to match.
    Models occasionally re-escape backslashes from a tool's JSON-rendered error
    message (e.g. shell escapes like ``\\(`` rendered as ``\\\\(`` in JSON, which
    the model copies back verbatim). Caller must guard with a uniqueness check
    to avoid corrupting content that legitimately uses ``\\\\``.
    """
    return s.replace("\\\\", "\\") if "\\\\" in s else s


# Fuzzy-match confidence tiers for a failed `old_str`. Above _FUZZY_STRONG the
# block is almost certainly the intended one; between the two it is offered
# with an explicit caveat.
_FUZZY_STRONG = 0.85
_FUZZY_WEAK = 0.6

def _requested_span(offset: int, limit: Any) -> tuple[int, int]:
    """(start, end) line range for a read. `end` is open-ended without a limit."""
    start = max(0, int(offset or 0))
    if isinstance(limit, int) and limit > 0:
        return start, start + limit
    return start, 1 << 30


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# Overlapping re-reads of the same region tolerated before read_file refuses.
_MAX_OVERLAPPING_READS = 4

_UNDO_STACK_CAP = 50
# Identical cached reads tolerated before read_file refuses (1 = the first
# repeat is served from cache, the next one errors).
_MAX_CACHE_HITS = 2


def _push_undo(ctx: ToolContext, abs_path: str, original: str | None) -> None:
    """Record a reversible mutation, bounding the stack so full pre-edit file
    contents don't accumulate for the whole process lifetime.

    ``original=None`` marks a newly-created file (undo removes it).
    """
    ctx.undo_stack.append((abs_path, original))
    if len(ctx.undo_stack) > _UNDO_STACK_CAP:
        del ctx.undo_stack[: len(ctx.undo_stack) - _UNDO_STACK_CAP]


def _invalidate_read_cache(ctx: ToolContext, abs_path: str) -> None:
    """Drop any cached reads for *abs_path* after a mutating write/edit."""
    for key in [k for k in ctx.files_read_meta if k[0] == abs_path]:
        del ctx.files_read_meta[key]
    # Drop repeat counters too: after a real change, re-reading is legitimate.
    for key in [k for k in ctx.read_cache_hits if k[0] == abs_path]:
        del ctx.read_cache_hits[key]
    ctx.files_read.pop(abs_path, None)
    # files_read is keyed by relative path, so also pop relative form.
    rel_path = os.path.relpath(abs_path, ctx.working_dir)
    ctx.files_read.pop(rel_path, None)


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
 
 
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _symbol_hint(ctx: ToolContext, abs_path: str, old_str: str | None) -> str:
    """Point a failed edit at the real line spans of the symbols it named.

    Returns "" when there is no graph or nothing recognisable, so the caller
    falls back to its generic advice rather than being blocked by a feature
    that happens not to be available here.
    """
    try:
        rel = os.path.relpath(abs_path, ctx.working_dir).replace(os.sep, "/")
    except ValueError:
        return ""
    if rel.startswith(".."):
        return ""
    from squishy.tools.graph import graph_for

    graph = graph_for(ctx)
    if graph is None:
        return ""
    node = graph.get(rel)
    if node is None:
        return ""
    wanted = set(_IDENT_RE.findall(old_str or ""))
    children = list(graph.children(rel))
    named = [c for c in children if c.name in wanted]
    for child in list(children):
        named += [m for m in graph.children(child.node_id) if m.name in wanted]
    if named:
        lines = "\n".join(
            f"  {c.name}: lines {c.lineno}-{c.end_lineno}"
            for c in sorted(named, key=lambda n: n.lineno)[:6]
        )
        return (
            f" The symbols your old_str names are at these exact lines in "
            f"{rel}:\n{lines}\n"
            f"Read one of those ranges with read_file(offset/limit) and copy "
            f"the text verbatim, including indentation."
        )
    if children:
        lines = ", ".join(
            f"{c.name} (L{c.lineno})"
            for c in sorted(children, key=lambda n: n.lineno)[:12]
        )
        return (
            f" Nothing in {rel} matches that text. It defines: {lines}. Read "
            f"the one you meant with read_file(offset/limit) first."
        )
    return ""


def _outline_for(ctx: ToolContext, abs_path: str) -> str:
    """The graph's symbol map for a file, or "" when it has none.

    Empty is the safe answer: a file the graph does not cover (not Python, or
    indexed before it was written) must fall back to a real read rather than
    being refused.
    """
    try:
        rel = os.path.relpath(abs_path, ctx.working_dir).replace(os.sep, "/")
    except ValueError:
        return ""
    if rel.startswith(".."):
        return ""
    from squishy.graph.query import file_outline
    from squishy.tools.graph import graph_for

    graph = graph_for(ctx)
    return file_outline(graph, rel) if graph is not None else ""


async def _read_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path") or args.get("file_path") or args.get("file")
    if not isinstance(path, str):
        return ToolResult(False, error="`path` is required (string)")
    abs_path, err, _scratch = _resolve_writable(path, ctx.working_dir)
    if err:
        return ToolResult(False, error=err)
    corrected = ""
    if not os.path.isfile(abs_path):
        # Reading is side-effect-free, so when exactly one candidate matches we
        # serve it and say so, rather than spending a turn on a correction
        # round-trip. Ambiguous or hopeless cases still error, with candidates.
        cands = (
            [] if os.path.isdir(abs_path)
            else _path_candidates(path, ctx.working_dir)
        )
        if len(cands) == 1:
            corrected = cands[0]
            abs_path, err = _safe_resolve(corrected, ctx.working_dir)
            if err:
                return ToolResult(False, error=err)
        else:
            return ToolResult(
                False, error=_not_found_error(path, ctx.working_dir, abs_path))

    try:
        offset = int(float(args.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    limit = args.get("limit")
    if limit is not None:
        try:
            limit = int(float(limit))
        except (TypeError, ValueError):
            limit = None

    # Duplicate-read dedup (checked BEFORE the hard cap so cached reads are
    # always returned without counting against the limit).
    cache_key = (abs_path, offset, limit)
    prior = ctx.files_read_meta.get(cache_key)
    if prior is not None:
        # Escalate on a loop: the first repeat is answered from cache, but
        # further identical reads return an ERROR. Serving unlimited successful
        # cache hits let weak models spin on the same call until the generic
        # loop detector killed the run (observed live: 8 identical reads).
        hits = ctx.read_cache_hits.get(cache_key, 0) + 1
        ctx.read_cache_hits[cache_key] = hits
        if hits >= _MAX_CACHE_HITS:
            return ToolResult(
                False,
                error=(
                    f"Refused: you have already read this exact range of '{path}' "
                    f"{hits + 1} times and the content has not changed. STOP reading. "
                    "Use what you already have: call `edit_file` to make your change, "
                    "`run_command` to test, `save_note` to record findings, or reply "
                    "with a plain-text summary if the task is done."
                ),
            )
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

    # Hard cap: refuse circling back over content the model already has.
    #
    # Only *overlapping* re-reads count. Every read reaching this point is a
    # range not served before (exact repeats returned from the cache above), so
    # counting all of them punished paging through a large file with
    # offset/limit — which is precisely what read_file's own description tells
    # the model to do. That made "read_file: refused" the top tool failure in
    # two consecutive sweeps (106, then 131).
    #
    # Files where edit_file just failed are exempt either way: the model needs
    # fresh content for an accurate old_str.
    span = _requested_span(offset, limit)
    prior_spans = ctx.files_read_spans.get(abs_path, [])
    redundant = sum(1 for s in prior_spans if _spans_overlap(s, span))
    if redundant >= _MAX_OVERLAPPING_READS and abs_path not in ctx.edit_fail_files:
        return ToolResult(
            False,
            error=(
                f"Refused: you have already read these lines of '{path}' "
                f"{redundant} times. You have the content — call `edit_file` "
                f"with your fix, or read a different part of the file with "
                f"offset/limit."
            ),
        )

    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return ToolResult(False, error=str(e))

    lines = text.splitlines()
    sliced = lines[offset : offset + limit] if limit is not None else lines[offset:]
    content = "\n".join(sliced)

    # A whole-file read of something too big to return: send the outline.
    #
    # Gated on "too big" for a reason that was measured the hard way. A turn
    # costs 8.5-11k prompt tokens because the whole transcript is resent, while
    # an outline saves 0.4k on a short file and 3.7k on a long one — so an
    # outline that forces a follow-up read is a net loss at every size.
    # Shipping it unconditionally took cfn-lint's graph arm from 20/29/42 tool
    # calls to 43/48/29 and cost it both its resolves. Above the output cap the
    # body gets snipped in the middle anyway, so the follow-up read was always
    # going to happen, and an outline beats a truncated prefix.
    outline = ""
    if offset == 0 and limit is None and len(content) > ctx.max_tool_output_chars:
        outline = _outline_for(ctx, abs_path)
    if outline:
        ctx.files_read_count[abs_path] = ctx.files_read_count.get(abs_path, 0) + 1
        return ToolResult(
            True,
            data={
                "path": path,
                "outline": outline,
                "total_lines": len(lines),
                "note": (
                    f"{path} is {len(lines)} lines — too long to return in one "
                    "call. This is its symbol map; read the range you need "
                    "with offset/limit"
                    + (
                        ", or call explore() on a symbol."
                        if profile_shows(ctx.tool_profile, "explore")
                        else "."
                    )
                ),
            },
            display=f"outline ({len(lines)} lines)",
        )

    ctx.files_read[path] = content
    ctx.files_read_meta[cache_key] = {
        "content": content,
        "total_lines": len(lines),
        "returned_lines": len(sliced),
    }
    # Track total reads per path (regardless of offset/limit) for the advisory
    # warning, and the concrete span served for the overlap-based hard cap.
    ctx.files_read_count[abs_path] = ctx.files_read_count.get(abs_path, 0) + 1
    path_reads = ctx.files_read_count[abs_path]
    served = (span[0], span[0] + len(sliced)) if sliced else span
    ctx.files_read_spans.setdefault(abs_path, []).append(served)

    data: dict[str, Any] = {
        "path": path,
        "content": content,
        "total_lines": len(lines),
        "returned_lines": len(sliced),
        "offset": offset,
    }
    if corrected:
        # Say what was actually read, so the model uses the right path next
        # time instead of repeating the miss on its edit call.
        data["path"] = corrected
        data["note"] = (
            f"'{path}' does not exist; read '{corrected}' instead. "
            f"Use '{corrected}' in your next call."
        )
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
    path = args.get("path") or args.get("file_path") or args.get("file")
    content = next((args[k] for k in ("content", "text", "data") if k in args and args[k] is not None), None)
    if not isinstance(path, str) or not isinstance(content, str):
        return ToolResult(False, error="`path` and `content` are required strings")
    abs_path, err, scratch = _resolve_writable(path, ctx.working_dir)
    if err:
        return ToolResult(False, error=err)

    if scratch:
        # Scratch files are throwaway by definition: they never reach the diff,
        # so the new-files-only rule, the test-file refusal and the
        # mistyped-path check below are all irrelevant, and enforcing them
        # would make the second draft of a repro script impossible.
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(
            True,
            data={"path": path, "bytes": len(content.encode()), "scratch": True},
            display=f"wrote {len(content.encode())} bytes to scratch",
        )

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

    # Bench mode: block creating test files and reproduction scripts.
    if ctx.permission_mode == "bench":
        basename = os.path.basename(abs_path).lower()
        rel = os.path.relpath(abs_path, ctx.working_dir)
        is_test_file = (
            basename.startswith("test_")
            or "/tests/" in rel or rel.startswith("tests/")
            or "repro" in basename or "reproduction" in basename
        )
        if is_test_file:
            # Refuse the location, not the technique. Reproducing the bug in a
            # throwaway script is how the failure gets confirmed at all — on
            # msrest-for-python-43 it is the entire difference between the
            # reference agent (nine repro snippets, resolved) and this one
            # (zero, sixty turns, no patch). The only real objection is that a
            # scratch file inside the repo lands in the graded diff, and /tmp
            # answers that. Refusing outright and saying "fix the SOURCE code
            # instead" withdrew the tool and the technique together.
            return ToolResult(
                False,
                error=(
                    f"write_file refused — {path} is inside the repo, so it would "
                    "land in the graded diff.\n"
                    "Write it under /tmp instead and run it from there:\n"
                    f'  write_file(path="/tmp/{os.path.basename(abs_path)}", content=...)\n'
                    '  run_command(command="python /tmp/'
                    f'{os.path.basename(abs_path)}")\n'
                    "The fix itself still belongs in the source file — use `edit_file` for that."
                ),
            )

    # A mistyped path here doesn't error — it silently creates a stray file
    # (e.g. `vyper/pyproject.toml` beside the real one) and the intended file
    # is never touched. If the name exists elsewhere, that's almost certainly
    # the target.
    misplaced = _path_candidates(path, ctx.working_dir)
    if misplaced:
        return ToolResult(
            False,
            error=(
                f"write_file refused — {path} does not exist, but "
                f"{', '.join(misplaced)} does. You probably meant that file; "
                f"use `edit_file` on it. If you really do want a new file at "
                f"{path}, say so by creating its directory first."
            ),
        )

    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Record the creation so undo_edit can remove the new file (previously
    # only edit_file was undoable, so an undo after write_file reverted the
    # wrong file).
    _push_undo(ctx, abs_path, None)
    _invalidate_read_cache(ctx, abs_path)
    encoded = content.encode("utf-8")
    return ToolResult(
        True,
        data={"path": path, "bytes": len(encoded)},
        display=f"wrote {len(encoded)} bytes",
    )
 
 
async def _edit_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    # Accept common aliases for parameter names (models sometimes use wrong names).
    _OLD_KEYS = ("old_str", "old_string", "old_text", "original", "search", "insert_after")
    _NEW_KEYS = ("new_str", "new_string", "new_text", "replacement", "replace", "new_lines")
    # v6e: full set of keys we recognize, used to flag silently-ignored params
    # in the missing-required-param error path below.
    _KNOWN_EDIT_KEYS = (
        {"path", "file_path", "file", "replace_all", "_tool_arg_error"}
        | set(_OLD_KEYS) | set(_NEW_KEYS)
    )
    path = args.get("path") or args.get("file_path") or args.get("file")
    old_str = next((args[k] for k in _OLD_KEYS if k in args and args[k] is not None), None)
    new_str = next((args[k] for k in _NEW_KEYS if k in args and args[k] is not None), None)
    replace_all = bool(args.get("replace_all", False))
    missing = []
    if not isinstance(path, str):
        missing.append("path")
    if not isinstance(old_str, str):
        missing.append("old_str")
    if not isinstance(new_str, str):
        missing.append("new_str")
    if missing:
        # v6e: also surface any unrecognized keys the caller sent — when a
        # model uses a different schema (e.g. {start_line, end_line, text}),
        # silently ignoring those keys leads to repeat failures on the next
        # turn.  Listing them tells the model exactly which params were
        # dropped so it can correct its mental model.
        unknown = sorted(set(args) - _KNOWN_EDIT_KEYS)
        extra_hint = (
            f" Unrecognized parameters (silently ignored): {', '.join(unknown)}."
            if unknown else ""
        )
        # Surface the accepted aliases so the model can self-correct on retry.
        return ToolResult(
            False,
            error=(
                f"Missing or non-string parameter(s): {', '.join(missing)}. "
                f"Required parameters are `path`, `old_str`, `new_str` (all strings). "
                f"Accepted aliases: path/file_path/file, "
                f"old_str/old_string/old_text/original/search, "
                f"new_str/new_string/new_text/replacement/replace."
                f"{extra_hint}"
            ),
        )
 
    # An empty old_str matches between every character; with replace_all it
    # would splice new_str throughout the file (corruption), and without it the
    # replacement is meaningless. Reject it outright.
    if old_str == "":
        return ToolResult(
            False,
            error=(
                "old_str must not be empty. To insert text, include an exact "
                "anchor snippet from the file in old_str and put the anchor plus "
                "your new text in new_str. To create a new file, use write_file."
            ),
        )

    abs_path, err, _scratch = _resolve_writable(path, ctx.working_dir)
    if err:
        return ToolResult(False, error=err)
    if not os.path.isfile(abs_path):
        # Unlike read_file, never auto-correct here: guessing wrong would
        # silently edit a file the model didn't ask for.
        return ToolResult(
            False, error=_not_found_error(path, ctx.working_dir, abs_path))

    with open(abs_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    def _save_undo() -> None:
        _push_undo(ctx, abs_path, text)

    count = text.count(old_str)
    if count == 0:
        # Stage 1: try stripping trailing whitespace from each line
        text_stripped = "\n".join(line.rstrip() for line in text.split("\n"))
        old_stripped = "\n".join(line.rstrip() for line in old_str.split("\n"))
        stripped_count = text_stripped.count(old_stripped)

        if stripped_count == 1:
            # Map match position from stripped text back to original lines
            # so only the matched region loses trailing whitespace.
            orig_lines = text.split("\n")

            # Determine which lines the match covers in the stripped text.
            pre_match = text_stripped[: text_stripped.index(old_stripped)]
            start_line = pre_match.count("\n")
            old_line_count = old_stripped.count("\n") + 1
            end_line = start_line + old_line_count

            # Unescape new_str if it contains literal escape sequences
            _new_str = _unescape_str(new_str) if "\\" in new_str else new_str
            new_stripped = "\n".join(line.rstrip() for line in _new_str.split("\n"))
            before = "\n".join(orig_lines[:start_line])
            after = "\n".join(orig_lines[end_line:])
            parts = [p for p in (before, new_stripped, after) if p]
            new_text = "\n".join(parts) if parts else ""
            # Preserve trailing newline if original had one
            if text.endswith("\n") and not new_text.endswith("\n"):
                new_text += "\n"
            _save_undo()
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            _invalidate_read_cache(ctx, abs_path)
            old_lines = len(old_str.splitlines()) or 1
            new_lines = len(_new_str.splitlines()) or 1
            return ToolResult(
                True,
                data={
                    "path": path,
                    "replacements": 1,
                    "old_lines": old_lines,
                    "new_lines": new_lines,
                    "old_str": old_str,
                    "new_str": _new_str,
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

        # Stage 1b: unescape over-escaped characters (e.g. \" → ", \n → newline)
        # Models sometimes double-escape JSON strings, producing literal backslash
        # sequences that don't match the actual file content.
        old_unesc = _unescape_str(old_str)
        if old_unesc != old_str:
            unesc_count = text.count(old_unesc)
            # Only unescape new_str if it also contains escape sequences —
            # otherwise the model sent intentional backslashes (e.g. regex
            # patterns, Windows paths) that must be preserved verbatim.
            new_unesc = _unescape_str(new_str) if "\\" in new_str else new_str
            if unesc_count == 1 or (unesc_count > 1 and replace_all):
                new_text = text.replace(old_unesc, new_unesc, -1 if replace_all else 1)
                _save_undo()
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                _invalidate_read_cache(ctx, abs_path)
                old_lines = len(old_unesc.splitlines()) or 1
                new_lines = len(new_unesc.splitlines()) or 1
                return ToolResult(
                    True,
                    data={
                        "path": path,
                        "replacements": unesc_count if replace_all else 1,
                        "old_lines": old_lines,
                        "new_lines": new_lines,
                        "old_str": old_unesc,
                        "new_str": new_unesc,
                        "note": "escape sequences normalized",
                    },
                    display=f"{old_lines} → {new_lines} lines (escape sequences normalized)",
                )
            if unesc_count > 1 and not replace_all:
                context_snippets = _collect_match_context(
                    text, old_unesc, max_matches=3, context_lines=2
                )
                return ToolResult(
                    False,
                    error=(
                        f"old_str matched {unesc_count} times after unescaping. "
                        f"Add more surrounding context or use replace_all=true. "
                        f"Match sites:\n" + context_snippets
                    ),
                )

        # Stage 1c: last-resort fallback — collapse literal `\\` → `\`. Models
        # sometimes re-escape backslashes copied from a JSON-rendered error
        # message (e.g. shell escapes like `\(` shown as `\\(` in tool output,
        # which the model echoes back verbatim). Guard with uniqueness check to
        # avoid corrupting content that legitimately contains `\\`.
        old_collapsed = _collapse_double_backslash(old_str)
        if old_collapsed != old_str:
            collapsed_count = text.count(old_collapsed)
            new_collapsed = (
                _collapse_double_backslash(new_str) if "\\\\" in new_str else new_str
            )
            if collapsed_count == 1 or (collapsed_count > 1 and replace_all):
                new_text = text.replace(
                    old_collapsed, new_collapsed, -1 if replace_all else 1
                )
                _save_undo()
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                _invalidate_read_cache(ctx, abs_path)
                old_lines = len(old_collapsed.splitlines()) or 1
                new_lines = len(new_collapsed.splitlines()) or 1
                return ToolResult(
                    True,
                    data={
                        "path": path,
                        "replacements": collapsed_count if replace_all else 1,
                        "old_lines": old_lines,
                        "new_lines": new_lines,
                        "old_str": old_collapsed,
                        "new_str": new_collapsed,
                        "note": "double-backslash collapsed",
                    },
                    display=f"{old_lines} → {new_lines} lines (double-backslash collapsed)",
                )
            if collapsed_count > 1 and not replace_all:
                context_snippets = _collect_match_context(
                    text, old_collapsed, max_matches=3, context_lines=2
                )
                return ToolResult(
                    False,
                    error=(
                        f"old_str matched {collapsed_count} times after collapsing "
                        f"double-backslashes. Add more surrounding context or use "
                        f"replace_all=true. Match sites:\n" + context_snippets
                    ),
                )

        # Stage 1d: combined fallback — collapse double-backslashes first, THEN
        # unescape the usual sequences. Catches double-JSON-encoded payloads
        # where the model sent both `\\"` (escaped quote) AND `\\\\` (escaped
        # backslash) AND `\\n` (escaped newline) in the same string. Stage 1b
        # alone leaves `\\\\` partially mangled; Stage 1c alone leaves `\\"`
        # untouched. Composing 1c → 1b in that order is the only fix.
        # Guard tightly with uniqueness check so we don't corrupt strings that
        # legitimately contain `\\` (e.g. Windows paths, regex patterns).
        old_combined = _unescape_str(_collapse_double_backslash(old_str))
        if old_combined != old_str and old_combined != _unescape_str(old_str) \
                and old_combined != _collapse_double_backslash(old_str):
            combined_count = text.count(old_combined)
            new_combined = (
                _unescape_str(_collapse_double_backslash(new_str))
                if "\\" in new_str else new_str
            )
            if combined_count == 1 or (combined_count > 1 and replace_all):
                new_text = text.replace(
                    old_combined, new_combined, -1 if replace_all else 1
                )
                _save_undo()
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                _invalidate_read_cache(ctx, abs_path)
                old_lines = len(old_combined.splitlines()) or 1
                new_lines = len(new_combined.splitlines()) or 1
                return ToolResult(
                    True,
                    data={
                        "path": path,
                        "replacements": combined_count if replace_all else 1,
                        "old_lines": old_lines,
                        "new_lines": new_lines,
                        "old_str": old_combined,
                        "new_str": new_combined,
                        "note": "double-encoded escapes normalized",
                    },
                    display=f"{old_lines} → {new_lines} lines (double-encoded escapes normalized)",
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

        # Stage 3: fuzzy match — find a contiguous block that closely matches
        # old_str. Single-line old_str is included: it is the most common shape
        # ("return 1") and excluding it sent the most frequent failure straight
        # to the useless "read the file first" fallback.
        if not hint and len(old_lines_list) >= 1:
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
            if best_ratio >= _FUZZY_WEAK and best_start >= 0:
                actual_lines = file_lines[best_start : best_start + num_old]
                actual_block = "\n".join(actual_lines)
                # Below the strong threshold this may well be the wrong block,
                # so say so rather than implying a confident location. Even a
                # tentative pointer beats "read the file first", which gives
                # the model nowhere to go.
                lead = (
                    " A similar block was found"
                    if best_ratio >= _FUZZY_STRONG
                    else " The closest block found (may not be the right one)"
                )
                hint = (
                    f"{lead} at lines {best_start + 1}-"
                    f"{best_start + num_old} ({best_ratio:.0%} match):\n"
                    f"---\n{actual_block}\n---\n"
                    f"Re-call edit_file with the exact text above as old_str."
                )

        if not hint:
            # Last resort, and the one that used to be a dead end. "Read the
            # file first" is advice the model has usually already taken:
            # qiskit-terra-5662's only edit attempt came at turn 73, after
            # five reads of that same file, and it never tried again. The
            # graph knows every symbol's exact span, so name them with their
            # line numbers instead of sending it back to guess.
            hint = _symbol_hint(ctx, abs_path, old_str) or (
                " Read the file first and copy the exact text."
            )

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
 
    # Proactively unescape new_str if it contains literal escape sequences
    # (e.g. \n, \t, \") that the model serialized instead of real characters.
    # Safe: old_str matched the real file, so backslash sequences in new_str
    # at function/class boundaries are clearly serialization artefacts.
    new_unesc = _unescape_str(new_str) if "\\" in new_str else new_str
    if new_unesc != new_str:
        new_str = new_unesc

    new_text = text.replace(old_str, new_str, -1 if replace_all else 1)
    _save_undo()
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(new_text)

    _invalidate_read_cache(ctx, abs_path)
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
    show_hidden = bool(args.get("show_hidden", False))
    if not isinstance(path, str):
        return ToolResult(False, error="`path` must be a string")
    abs_path, err = _safe_resolve(path, ctx.working_dir)
    if err:
        return ToolResult(False, error=err)
    if not os.path.isdir(abs_path):
        return ToolResult(False, error=_not_a_dir_error(path, ctx.working_dir, abs_path))

    entries = []
    for name in sorted(os.listdir(abs_path)):
        if name in SKIP_DIRS:
            continue
        if not show_hidden and name.startswith("."):
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
 
    abs_path, err = _safe_resolve(
        search_path if isinstance(search_path, str) else ".", ctx.working_dir
    )
    if err:
        return ToolResult(False, error=err)
    if not os.path.exists(abs_path):
        return ToolResult(False, error=f"path not found: {search_path}")
 
    cap = 50 if ctx.permission_mode == "bench" else SEARCH_CAP
    rg = shutil.which("rg")

    async def _run(pat: str, *, force_literal: bool = False) -> ToolResult:
        if rg:
            return await _rg_search(rg, pat, abs_path, glob, cap=cap,
                                    force_literal=force_literal)
        return await asyncio.to_thread(_python_search, pat, abs_path, glob,
                                       cap=cap, force_literal=force_literal)

    res = await _run(pattern)

    # A pattern can be valid regex and still not mean what the model intended:
    # `data[0]` is a character class, so searching for the literal text
    # `data[0]` returns nothing and the model concludes the code isn't there.
    # When a metacharacter-bearing pattern finds nothing, retry it literally.
    if (
        res.success
        and not res.data.get("count")
        and _HAS_REGEX_META.search(pattern)
    ):
        literal_res = await _run(pattern, force_literal=True)
        if literal_res.success and literal_res.data.get("count"):
            literal_res.data["note"] = (
                f"No regex matches for '{pattern}', but it appears literally "
                f"in the code — showing literal matches."
            )
            return literal_res
    return res
 
 
# Characters that make a pattern "look like" a regex. Used to decide
# whether a zero-match search is worth retrying as a literal string.
_HAS_REGEX_META = re.compile(r"[\\\[\]().*+?{}|^$]")


def _is_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


async def _rg_search(
    rg: str, pattern: str, abs_path: str, glob: Any, *, cap: int = SEARCH_CAP,
    force_literal: bool = False,
) -> ToolResult:
    # Models routinely pass a glob (`*.py`) or a raw code snippet where a
    # regex is expected. Searching those literally is what they meant; failing
    # is never useful.
    literal = force_literal or not _is_regex(pattern)
    cmd = [rg, "-n", "--no-heading", "-S"]
    if literal:
        cmd.append("-F")
    cmd += [pattern, abs_path]
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
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=SEARCH_TIMEOUT)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(False, error="ripgrep timed out")
    except FileNotFoundError as e:
        return ToolResult(False, error=str(e))

    # rg exits 1 for "no matches" and >=2 for a real error. Ignoring the code
    # reported a broken search as "0 matches", so the model concluded the
    # symbol didn't exist and moved on.
    if proc.returncode is not None and proc.returncode >= 2:
        msg = stderr.decode("utf-8", errors="replace").strip().splitlines()
        return ToolResult(
            False,
            error=f"search failed: {msg[0] if msg else 'ripgrep error'}",
        )
 
    matches: list[dict[str, Any]] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines()[:cap]:
        parts = line.split(":", 2)
        if len(parts) == 3:
            matches.append({"file": parts[0], "line": int(parts[1]), "text": parts[2]})
    data: dict[str, Any] = {
        "pattern": pattern, "matches": matches, "count": len(matches),
    }
    if literal:
        data["note"] = (
            f"'{pattern}' is not a valid regex; searched for it literally."
        )
    return ToolResult(
        True,
        data=data,
        display=f"{len(matches)} matches" + (" (literal)" if literal else ""),
    )
 
 
def _python_search(pattern: str, abs_path: str, glob: Any, *, cap: int = SEARCH_CAP,
                   force_literal: bool = False) -> ToolResult:
    # Same fallback as the ripgrep path: an unparseable pattern is treated as
    # a literal string rather than rejected.
    literal = force_literal
    if force_literal:
        rx = re.compile(re.escape(pattern))
    else:
        try:
            rx = re.compile(pattern)
        except re.error:
            literal = True
            rx = re.compile(re.escape(pattern))
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
 
    data: dict[str, Any] = {
        "pattern": pattern, "matches": matches, "count": len(matches),
    }
    if literal:
        data["note"] = (
            f"'{pattern}' is not a valid regex; searched for it literally."
        )
    return ToolResult(
        True,
        data=data,
        display=f"{len(matches)} matches" + (" (literal)" if literal else ""),
    )


read_file = Tool(
    name="read_file",
    description=(
        "Read a file and return its contents with a line count. "
        "Repeating the identical read is refused — page through a large file "
        "with offset/limit instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the working directory, e.g. 'src/app.py'.",
            },
            "offset": {
                "type": "integer", "default": 0,
                "description": "First line to return, 0-based.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return. Omit to read to the end.",
            },
        },
        "required": ["path"],
    },
    run=_read_file,
)
 
write_file = Tool(
    name="write_file",
    description=(
        "Create a NEW file. Refused if the path already exists — use edit_file "
        "to change an existing file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path for the new file, relative to the working directory.",
            },
            "content": {
                "type": "string",
                "description": "Complete contents of the file.",
            },
        },
        "required": ["path", "content"],
    },
    run=_write_file,
)
 
edit_file = Tool(
    name="edit_file",
    description=(
        "Change an existing file by replacing an exact snippet of its text. "
        "This is the way to modify code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the working directory.",
            },
            "old_str": {
                "type": "string",
                "description": (
                    "Text to replace, copied verbatim from the file including "
                    "indentation. Must appear exactly once unless replace_all "
                    "is true; add surrounding lines to make it unique."
                ),
            },
            "new_str": {
                "type": "string",
                "description": "Text to put in its place. Use \"\" to delete.",
            },
            "replace_all": {
                "type": "boolean", "default": False,
                "description": "Replace every occurrence instead of requiring a unique match.",
            },
        },
        "required": ["path", "old_str", "new_str"],
    },
    run=_edit_file,
)
 
list_directory = Tool(
    name="list_directory",
    description="List files and directories. Hides dotfiles by default (use show_hidden=true to include them).",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string", "default": ".",
                "description": "Directory to list, relative to the working directory.",
            },
            "show_hidden": {
                "type": "boolean",
                "default": False,
                "description": "Include dotfiles/hidden files in the listing",
            },
        },
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
            "path": {
                "type": "string", "default": ".",
                "description": "Directory or file to search under, relative to the working directory.",
            },
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
        abs_path, err = _safe_resolve(path, ctx.working_dir)
        if err:
            return ToolResult(False, error=err)
        cmd += ["--", abs_path]

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
    description="Show git diff of your uncommitted changes. Pass a path or omit for all.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File to diff (omit for all)"},
        },
    },
    run=_show_diff,
)


GLOB_CAP = 200


def _glob_files_sync(abs_path: str, pattern: str) -> list[str]:
    """Synchronous glob helper — runs in a thread to avoid blocking the loop."""
    from pathlib import Path

    base = Path(abs_path)
    matches: list[str] = []
    for p in sorted(base.glob(pattern)):
        parts = p.relative_to(base).parts
        if any(part.startswith(".") or part in SKIP_DIRS for part in parts):
            continue
        if p.is_file():
            matches.append(str(p.relative_to(base)))
        if len(matches) >= GLOB_CAP:
            break
    return matches


async def _glob_files(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Find files matching a glob pattern recursively."""
    pattern = args.get("pattern")
    if not isinstance(pattern, str):
        return ToolResult(False, error="`pattern` is required (string glob)")
    path = args.get("path", ".")
    abs_path, err = _safe_resolve(path if isinstance(path, str) else ".", ctx.working_dir)
    if err:
        return ToolResult(False, error=err)
    if not os.path.isdir(abs_path):
        return ToolResult(False, error=_not_a_dir_error(path, ctx.working_dir, abs_path))

    try:
        matches = await asyncio.to_thread(_glob_files_sync, abs_path, pattern)
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


async def _undo_edit(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if not ctx.undo_stack:
        return ToolResult(False, error="Nothing to undo. No edits have been made yet.")
    abs_path, original = ctx.undo_stack.pop()
    rel_path = os.path.relpath(abs_path, ctx.working_dir)
    if original is None:
        # The entry records a file created by write_file — undo removes it.
        try:
            os.remove(abs_path)
        except FileNotFoundError:
            pass
        except OSError as e:
            return ToolResult(False, error=f"could not remove {rel_path}: {e}")
        _invalidate_read_cache(ctx, abs_path)
        return ToolResult(
            True,
            data={"path": rel_path, "removed": True},
            display=f"removed {rel_path} (undo of write_file)",
        )
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(original)
    _invalidate_read_cache(ctx, abs_path)
    return ToolResult(
        True,
        data={"path": rel_path, "reverted_bytes": len(original.encode("utf-8"))},
        display=f"reverted {rel_path}",
    )


undo_edit = Tool(
    name="undo_edit",
    description="Revert the most recent edit_file/write_file change (LIFO; repeatable).",
    parameters={"type": "object", "properties": {}},
    run=_undo_edit,
)


FS_TOOLS: list[Tool] = [read_file, write_file, edit_file, undo_edit, list_directory, search_files, show_diff, glob_files]