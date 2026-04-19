"""Index optimization utilities.

Provides pruning and compaction utilities to keep the index small and fresh.
Includes:
- Configurable retention policies
- Stale summary detection and cleanup
- Directory pruning based on file activity
"""

from __future__ import annotations

import time
from pathlib import Path

from squishy.index.model import Index, Node


# Default retention period: prune summaries older than this (30 days)
DEFAULT_RETENTION_DAYS = 30
# Minimum files per dir to keep it
MIN_DIR_FILES = 1


def prune_stale_summaries(index: Index, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Prune summaries older than retention period.

    Returns count of summaries removed.
    """
    now = time.time()
    cutoff = now - (retention_days * 24 * 60 * 60)
    removed = 0

    for node in index.root.walk():
        if node.last_summarized > 0 and node.last_summarized < cutoff:
            node.summary = ""
            node.summary_hash = ""
            removed += 1

    if removed > 0:
        index.meta.summary_stats["pruned_summaries"] = removed

    return removed


def prune_empty_dirs(index: Index, min_files: int = MIN_DIR_FILES) -> int:
    """Remove directory nodes with fewer than min_files descendants.

    Returns count of dirs removed.
    """
    removed = 0

    def _should_prune(node: Node) -> bool:
        if node.kind != "dir":
            return False
        file_count = sum(1 for n in node.walk() if n.kind == "file")
        return file_count < min_files

    def _collect_prunable(node: Node, parent: Node | None = None) -> list[tuple[Node, Node]]:
        prunable: list[tuple[Node, Node]] = []
        for child in node.children[:]:
            prunable.extend(_collect_prunable(child, node))
            if _should_prune(child):
                prunable.append((child, node))
        return prunable

    # Collect all prunable dirs (bottom-up)
    prunable = _collect_prunable(index.root)

    # Remove from parent to avoid index issues
    for child, parent in prunable:
        if child in parent.children:
            parent.children.remove(child)
            removed += 1

    return removed


def compact_dir_summaries(index: Index) -> int:
    """Compact directory summaries by removing redundant info.

    If a dir's summary is just "This directory contains X, Y, Z",
    replace with shorter placeholder.

    Returns count of compacted summaries.
    """
    COMPACT_PATTERNS = [
        "This directory contains",
        "The directory contains",
        "Contains files and",
    ]

    compacted = 0
    for node in index.root.walk():
        if node.kind != "dir" or not node.summary:
            continue

        summary_lower = node.summary.lower()
        for pattern in COMPACT_PATTERNS:
            if summary_lower.startswith(pattern.lower()):
                # Replace with shorter form
                rest = node.summary[len(pattern) :].strip()
                if len(rest) > 50:  # Only compact long ones
                    node.summary = f"Dir: {rest[:47]}..."
                    compacted += 1
                break

    return compacted


def get_index_size(index: Index) -> dict[str, int]:
    """Get index size metrics."""
    file_count = 0
    dir_count = 0
    symbol_count = 0
    total_chars = 0

    for node in index.root.walk():
        if node.kind == "file":
            file_count += 1
        elif node.kind == "dir":
            dir_count += 1
        elif node.kind in ("class", "function", "method"):
            symbol_count += 1

        total_chars += len(node.name) + len(node.path)
        if node.summary:
            total_chars += len(node.summary)

    return {
        "files": file_count,
        "dirs": dir_count,
        "symbols": symbol_count,
        "total_nodes": file_count + dir_count + symbol_count,
        "estimated_chars": total_chars,
    }


def needs_pruning(index: Index, max_chars: int = 50_000) -> bool:
    """Check if index needs pruning based on size."""
    size = get_index_size(index)
    return size["estimated_chars"] > max_chars


def optimize_for_size(index: Index, max_chars: int = 50_000) -> dict[str, int]:
    """Apply size optimizations until under limit.

    Returns optimization stats.
    """
    stats: dict[str, int] = {
        "pruned_summaries": 0,
        "pruned_dirs": 0,
        "compacted_summaries": 0,
    }

    while needs_pruning(index, max_chars):
        initial_size = get_index_size(index)["estimated_chars"]

        # Try different strategies
        stats["pruned_summaries"] += prune_stale_summaries(index, retention_days=7)
        stats["pruned_dirs"] += prune_empty_dirs(index, min_files=1)
        stats["compacted_summaries"] += compact_dir_summaries(index)

        new_size = get_index_size(index)["estimated_chars"]
        if new_size >= initial_size:
            break  # No more progress possible

    return stats


__all__ = [
    "prune_stale_summaries",
    "prune_empty_dirs",
    "compact_dir_summaries",
    "get_index_size",
    "needs_pruning",
    "optimize_for_size",
    "DEFAULT_RETENTION_DAYS",
    "MIN_DIR_FILES",
]
