"""Repo indexer. See `squishy/index/build.py` and the `/init` command."""

from __future__ import annotations

from squishy.index.agents_md import generate_agents_md, save_agents_md
from squishy.index.build import _build_index_async, build_index
from squishy.index.model import Index, IndexMeta, Node
from squishy.index.optimize import (
    MIN_DIR_FILES,
    DEFAULT_RETENTION_DAYS,
    compact_dir_summaries,
    get_index_size,
    needs_pruning,
    optimize_for_size,
    prune_empty_dirs,
    prune_stale_summaries,
)
from squishy.index.staleness import describe_deep_staleness, describe_staleness
from squishy.index.store import (
    index_dir,
    index_path,
    load_index,
    load_meta,
    meta_path,
    save_index,
)
from squishy.index.summarize import Summarizer

__all__ = [
    "Index",
    "IndexMeta",
    "Node",
    "Summarizer",
    "_build_index_async",
    "build_index",
    "compact_dir_summaries",
    "describe_deep_staleness",
    "describe_staleness",
    "generate_agents_md",
    "get_index_size",
    "index_dir",
    "index_path",
    "load_index",
    "load_meta",
    "meta_path",
    "needs_pruning",
    "optimize_for_size",
    "prune_empty_dirs",
    "prune_stale_summaries",
    "save_agents_md",
    "save_index",
    # Constants
    "DEFAULT_RETENTION_DAYS",
    "MIN_DIR_FILES",
]
