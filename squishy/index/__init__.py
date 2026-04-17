"""Repo indexer. See `squishy/index/build.py` and the `/init` command."""
 
from __future__ import annotations
 
from squishy.index.build import build_index
from squishy.index.model import Index, IndexMeta, Node
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
    "build_index",
    "index_dir",
    "index_path",
    "load_index",
    "load_meta",
    "meta_path",
    "save_index",
]
