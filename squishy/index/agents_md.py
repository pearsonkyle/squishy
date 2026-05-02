"""Generate AGENTS.md from repo index.

Creates a human-readable project overview with:
- Project structure (tree visualization)
- Language distribution
- Key classes and functions with summaries
- Import/dependency hints (Python-specific)
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from squishy.index.model import Index, Node

# Don't include these in AGENTS.md (too noisy)
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "dist", "build"}
SKIP_FILES = {".gitignore", ".dockerignore", "Dockerfile"}


def _is_skip_dir(path: str) -> bool:
    """Check if path should be skipped."""
    parts = path.split("/")
    return any(p in SKIP_DIRS for p in parts)


def _is_skip_file(path: str) -> bool:
    """Check if file should be skipped."""
    basename = os.path.basename(path)
    return basename in SKIP_FILES or basename.startswith(".")


def _format_tree(root: Node, prefix: str = "", is_last: bool = True) -> list[str]:
    """Format the tree as a visual tree structure."""
    lines: list[str] = []
    connector = "└── " if is_last else "├── "

    # Format current node
    if root.kind == "repo":
        lines.append(f"{prefix}{'└── ' if prefix else ''}{root.name}/")
    elif root.kind == "dir":
        lines.append(f"{prefix}{connector}{root.name}/")
    elif root.kind == "file":
        ext = os.path.splitext(root.name)[1]
        icon = {"py": "🐍", "js": "🟨", "ts": "🔵", "go": "(go)", "rs": "🦀"}.get(
            ext.lstrip("."), "📄"
        )
        lines.append(f"{prefix}{connector}{icon} {root.name}")

    # Process children
    children = root.children or []
    for i, child in enumerate(sorted(children, key=lambda n: (n.kind != "dir", n.name))):
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
    """Get top directories by file count."""
    dir_counts: list[tuple[str, int]] = []
    for node in index.root.walk():
        if node.kind == "dir" and node.path:
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


def _generate_python_imports(index: Index, cwd: str) -> dict[str, list[str]]:
    """Extract import relationships for Python files."""
    imports: dict[str, list[str]] = defaultdict(list)

    for node in index.root.walk():
        if node.kind != "file" or not node.path.endswith(".py"):
            continue

        # Read file content
        abs_path = os.path.join(cwd, node.path)
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue

        # Simple import detection (not AST-level accurate, but good enough)
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                # Extract module name
                parts = line.split()
                if len(parts) >= 2:
                    mod = parts[1].split(".")[0]
                    imports[node.path].append(line)

    return dict(imports)


def _extract_summary(node: Node) -> str:
    """Get a summary for a node, preferring longer ones."""
    if node.summary:
        return node.summary[:200]
    # Fall back to first line of name as placeholder
    return f"{node.kind} {node.name}"


def generate_agents_md(index: Index, *, include_imports: bool = True, cwd: str = "") -> str:
    """Generate AGENTS.md content from an index.

    Args:
        index: The repo index to document
        include_imports: Include Python import relationships (default True)
        cwd: Working directory for resolving file paths (defaults to os.getcwd())

    Returns:
        Markdown content for AGENTS.md
    """
    if not cwd:
        cwd = os.getcwd()
    lines: list[str] = []

    # Header
    lines.append("# AGENTS.md")
    lines.append("")

    # Project stats
    file_count = sum(1 for n in index.root.walk() if n.kind == "file")
    symbol_count = sum(
        1 for n in index.root.walk() if n.kind in ("class", "function", "method")
    )
    lang_stats = _get_language_stats(index)
    lang_summary = ", ".join(
        f"{count} {ext}" for ext, count in sorted(lang_stats.items(), key=lambda kv: -kv[1])[:4]
    )
    lines.append(f"**{file_count} files, {symbol_count} symbols** ({lang_summary})")
    lines.append("")

    # Top-level directories (compact)
    top_dirs = _get_top_level_dirs(index)
    if top_dirs:
        lines.append("## Top Directories")
        lines.append("")
        for path, count in top_dirs:
            dir_name = os.path.basename(path) or "."
            # Include dir summary if available
            dir_node = next((n for n in index.root.walk() if n.kind == "dir" and n.path == path), None)
            summary = f" — {dir_node.summary}" if dir_node and dir_node.summary else ""
            lines.append(f"- **{dir_name}/**: {count} files{summary}")
        lines.append("")

    # Key symbols — most valuable section, placed early
    key_symbols = _extract_key_symbols(index)
    if key_symbols:
        lines.append("## Key Symbols")
        lines.append("")
        current_file = ""
        for sym in key_symbols:
            if sym["path"] != current_file:
                current_file = sym["path"]
                lines.append(f"### `{current_file}`")
            kind = sym["kind"]
            name = sym["name"]
            summary = sym["summary"]
            lines.append(f"- **{kind} `{name}`**: {summary}")
        lines.append("")

    # File summaries — compact list of files with docstring summaries
    file_summaries: list[tuple[str, str]] = []
    for node in index.root.walk():
        if node.kind == "file" and node.summary:
            file_summaries.append((node.path, node.summary))
    if file_summaries:
        lines.append("## File Summaries")
        lines.append("")
        for path, summary in sorted(file_summaries)[:40]:
            lines.append(f"- `{path}`: {summary}")
        lines.append("")

    # Directory structure (tree view) — only for small repos
    if file_count <= 100:
        lines.append("## Structure")
        lines.append("")
        tree_lines = _format_tree(index.root)
        for line in tree_lines:
            lines.append(line)
        lines.append("")

    # Navigation workflow
    lines.append("## Navigation")
    lines.append("")
    lines.append("Use `recall(query=...)` to search the index before calling `read_file`.")
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
