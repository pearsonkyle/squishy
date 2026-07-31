"""Generate AGENTS.md from repo index.

Creates a compact project overview for AI assistants. Prefers plain
prose / lists over bold/italics and avoids per-line code fences so
the file doesn't gratuitously eat context window when re-injected
as part of the system prompt.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from squishy.index.model import Index, Node

# Directories that are noise in the project overview: VCS internals,
# build artifacts, virtualenvs, dependency caches, and squishy's own
# generated session/index storage.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".idea", ".vscode", "sessions", ".squishy",
}
SKIP_FILES = {".gitignore", ".dockerignore", "Dockerfile"}

# Modules that ship with Python; we use stdlib_module_names when available
# (Python 3.10+) and fall back to a small hand-rolled set otherwise.
_STDLIB: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", ()) or {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "enum", "errno", "fnmatch",
    "functools", "glob", "gzip", "hashlib", "hmac", "html", "http", "io",
    "ipaddress", "itertools", "json", "logging", "math", "mimetypes",
    "operator", "os", "pathlib", "pickle", "pkgutil", "platform", "queue",
    "random", "re", "secrets", "shlex", "shutil", "signal", "socket",
    "sqlite3", "ssl", "string", "subprocess", "sys", "tempfile", "textwrap",
    "threading", "time", "traceback", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "warnings", "weakref", "xml", "zipfile",
})

# Top-level import patterns: "import foo", "from foo import ..." (capture foo's
# leading component before any dot).
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)")


def _is_skip_dir(path: str) -> bool:
    """Check if path should be skipped."""
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


def _is_skip_file(path: str) -> bool:
    """Check if file should be skipped."""
    basename = os.path.basename(path)
    return basename in SKIP_FILES or basename.startswith(".")


def _should_skip(node: Node) -> bool:
    """Filter for the AGENTS.md tree only — does not affect indexing."""
    if node.kind == "dir" and node.name in SKIP_DIRS:
        return True
    if node.kind == "file" and _is_skip_file(node.name):
        return True
    return False


def _format_tree(root: Node, prefix: str = "", is_last: bool = True) -> list[str]:
    """Format the tree as a visual tree structure.

    Plain names only — file-type emojis (🐍 📄 🦀 …) were dropped because
    every emoji costs 4+ bytes in the system prompt for zero signal that
    the extension wasn't already telling the model.
    """
    lines: list[str] = []
    connector = "└── " if is_last else "├── "

    if root.kind == "repo":
        lines.append(f"{prefix}{'└── ' if prefix else ''}{root.name}/")
    elif root.kind == "dir":
        lines.append(f"{prefix}{connector}{root.name}/")
    elif root.kind == "file":
        lines.append(f"{prefix}{connector}{root.name}")

    children = [c for c in (root.children or []) if not _should_skip(c)]
    children.sort(key=lambda n: (n.kind != "dir", n.name))
    for i, child in enumerate(children):
        is_child_last = i == len(children) - 1
        if root.kind == "repo":
            new_prefix = ""
        else:
            new_prefix = f"{prefix}{'    ' if is_last else '│   '}"

        lines.extend(_format_tree(child, new_prefix, is_child_last))

    return lines


def _get_language_stats(index: Index) -> dict[str, int]:
    """Count files by extension."""
    stats: dict[str, int] = defaultdict(int)
    for node in index.root.walk():
        if node.kind == "file":
            ext = os.path.splitext(node.name)[1].lower()
            if ext:
                stats[ext] += 1
    return dict(stats)


def _get_top_level_dirs(index: Index, limit: int = 5) -> list[tuple[str, int]]:
    """Get top directories by file count, skipping noise dirs."""
    dir_counts: list[tuple[str, int]] = []
    for node in index.root.walk():
        if node.kind == "dir" and node.path:
            if any(part in SKIP_DIRS for part in node.path.split("/")):
                continue
            n = sum(1 for c in node.walk() if c.kind == "file")
            dir_counts.append((node.path, n))
    return sorted(dir_counts, key=lambda kv: -kv[1])[:limit]


def _extract_key_symbols(index: Index, limit_per_file: int = 3) -> list[dict]:
    """Extract key classes and functions with their summaries."""
    symbols: list[dict] = []

    # Build a {path: file_node} lookup to avoid O(N^2) walk
    file_nodes: dict[str, Node] = {
        n.path: n for n in index.root.walk() if n.kind == "file"
    }

    for node in index.root.walk():
        if node.kind not in ("class", "function", "method"):
            continue
        if not node.summary:
            continue

        # Get containing file info via exact path match
        file_node = file_nodes.get(node.path)

        symbols.append({
            "name": node.name,
            "kind": node.kind,
            "path": node.path or (file_node.path if file_node else ""),
            "summary": node.summary[:200],
        })

    # Sort by importance: classes > functions > methods, then alphabetically
    kind_order = {"class": 0, "function": 1, "method": 2}
    symbols.sort(key=lambda s: (kind_order.get(s["kind"], 3), s["path"], s["name"]))

    # Limit per-file
    by_file: dict[str, list[dict]] = defaultdict(list)
    for s in symbols:
        by_file[s["path"]].append(s)

    result: list[dict] = []
    for path in sorted(by_file.keys()):
        file_syms = by_file[path][:limit_per_file]
        result.extend(file_syms)

    return result[:50]  # Overall limit


def _collect_external_deps(
    index: Index, cwd: str, *, top_n: int = 20,
) -> list[tuple[str, int]]:
    """Return (module, file_count) for non-stdlib, non-self imports.

    Replaces the previous per-file `import` dump, which repeated common
    lines (`import json`, `import os`, etc.) in dozens of code fences
    and chewed through the context window for almost no signal.
    """
    repo_pkg = (index.root.name or "").strip("/").split("/", 1)[0]
    counter: Counter[str] = Counter()
    for node in index.root.walk():
        if node.kind != "file" or not node.path.endswith(".py"):
            continue
        abs_path = os.path.join(cwd, node.path)
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        seen_in_file: set[str] = set()
        for raw in content.splitlines():
            m = _IMPORT_RE.match(raw)
            if not m:
                continue
            top = m.group(1).split(".", 1)[0]
            if not top or top in _STDLIB:
                continue
            if top in {repo_pkg, "squishy"}:  # ignore self-imports
                continue
            if top.startswith("_"):
                continue
            seen_in_file.add(top)
        for mod in seen_in_file:
            counter[mod] += 1
    return counter.most_common(top_n)


def _extract_summary(node: Node) -> str:
    """Get a summary for a node, preferring longer ones."""
    if node.summary:
        return node.summary[:200]
    # Fall back to first line of name as placeholder
    return f"{node.kind} {node.name}"


def _format_symbol(kind: str, name: str) -> str:
    """Render a symbol header without the redundant "function"/"method" word.

    - classes get a leading ``class `` prefix (the kind matters semantically)
    - functions and methods both get a trailing ``()`` and no prefix
    - anything else falls back to plain name
    """
    if kind == "class":
        return f"class {name}"
    if kind in ("function", "method"):
        return f"{name}()"
    return name


def generate_agents_md(index: Index, *, include_imports: bool = True, cwd: str = "") -> str:
    """Generate AGENTS.md content from an index.

    Args:
        index: The repo index to document
        include_imports: Emit a deduped external-deps list (default True).
        cwd: Working directory for resolving file paths (defaults to os.getcwd())

    Returns:
        Markdown content for AGENTS.md
    """
    if not cwd:
        cwd = os.getcwd()
    lines: list[str] = []

    # Header — plain prose, no horizontal rules / italics. The file is fed
    # back into the system prompt via load_agent_instructions, so every
    # decoration costs context budget.
    lines.append("# AGENTS.md")
    lines.append("")
    lines.append("Auto-generated by squishy /init. Project structure overview for AI assistants.")
    lines.append("")

    # Project stats — combine file/symbol counts with the language
    # distribution onto one line so a 6-language repo costs 1 line not 8.
    file_count = sum(1 for n in index.root.walk() if n.kind == "file")
    symbol_count = sum(
        1 for n in index.root.walk() if n.kind in ("class", "function", "method")
    )
    lang_stats = _get_language_stats(index)
    lang_str = ", ".join(
        f"{ext} {count}" for ext, count in sorted(lang_stats.items(), key=lambda kv: -kv[1])
    )
    summary_line = f"Files: {file_count}  Symbols: {symbol_count}"
    if lang_str:
        summary_line += f"  Langs: {lang_str}"
    lines.append(summary_line)
    lines.append("")

    # Directory structure (tree view) — usually the largest section by
    # far, but it's also where the model gets the most signal per byte.
    lines.append("## Structure")
    lines.append("")
    lines.extend(_format_tree(index.root))
    lines.append("")

    # Top-level directories — single line, no bullets.
    top_dirs = _get_top_level_dirs(index)
    if top_dirs:
        parts = [
            f"{(os.path.basename(path) or '.')}/ {count}"
            for path, count in top_dirs
        ]
        lines.append("Top dirs: " + ", ".join(parts))
        lines.append("")

    # Key symbols — drop the per-symbol "function"/"method" keyword
    # spam (the model already knows from the () suffix or `class `
    # prefix), drop the `### path` markdown header per file, and drop
    # the "method " label entirely (methods just look like functions
    # under their containing file's section).
    key_symbols = _extract_key_symbols(index)
    if key_symbols:
        lines.append("## Key symbols")
        lines.append("")
        current_file = ""
        for sym in key_symbols:
            if sym["path"] != current_file:
                current_file = sym["path"]
                if lines[-1] != "":
                    lines.append("")
                lines.append(f"{current_file}:")
            label = _format_symbol(sym["kind"], sym["name"])
            lines.append(f"- {label} — {sym['summary']}")
        lines.append("")

    # Replace the per-file imports dump with one deduped line of the most
    # common external (non-stdlib, non-self) deps. This used to repeat
    # `import json`, `import os` across every file in its own code fence.
    if include_imports:
        has_py = any(
            n.kind == "file" and n.path.endswith(".py") for n in index.root.walk()
        )
        if has_py:
            deps = _collect_external_deps(index, cwd)
            if deps:
                lines.append("## External deps")
                lines.append(", ".join(name for name, _ in deps))
                lines.append("")

    # Planning workflow — minimal markup, no nested bold.
    lines.append("## Planning workflow")
    lines.append("")
    lines.append("In plan mode:")
    lines.append("")
    lines.append("1. recall(query=...) first — use the index to find relevant files")
    lines.append("2. 1-2 targeted reads to understand the problem")
    lines.append("3. plan_task(problem=..., solution=..., steps=[...])")
    lines.append("")
    lines.append(
        "Prefer recall(query=...) to locate code fast. If the index misses or "
        "returns nothing useful, fall back to search_files / glob_files / "
        "read_file — exploration is never blocked. The index lives at "
        ".squishy/index.json."
    )
    lines.append("")


    return "\n".join(lines)


def save_agents_md(index: Index, cwd: str | os.PathLike[str]) -> Path:
    """Generate and save AGENTS.md to .squishy/ directory.

    Args:
        index: The repo index
        cwd: Working directory (where .squishy/ lives)

    Returns:
        Path to saved file
    """
    from squishy.index.store import index_dir

    content = generate_agents_md(index, cwd=str(cwd))
    agents_path = index_dir(cwd) / "AGENTS.md"
    agents_path.write_text(content, encoding="utf-8")
    return agents_path


__all__ = ["generate_agents_md", "save_agents_md"]
