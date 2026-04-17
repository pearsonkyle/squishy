"""Persist and load `.squishy/index.json` + `.squishy/index.meta.json`."""
 
from __future__ import annotations
 
import json
import os
from pathlib import Path
 
from squishy.index.model import Index, IndexMeta
 
INDEX_DIR = ".squishy"
INDEX_FILE = "index.json"
META_FILE = "index.meta.json"
 
 
def index_dir(cwd: str | os.PathLike[str]) -> Path:
    return Path(cwd) / INDEX_DIR
 
 
def index_path(cwd: str | os.PathLike[str]) -> Path:
    return index_dir(cwd) / INDEX_FILE
 
 
def meta_path(cwd: str | os.PathLike[str]) -> Path:
    return index_dir(cwd) / META_FILE
 
 
def save_index(cwd: str | os.PathLike[str], index: Index) -> Path:
    d = index_dir(cwd)
    d.mkdir(parents=True, exist_ok=True)
    ip = d / INDEX_FILE
    mp = d / META_FILE
    ip.write_text(json.dumps(index.to_dict(), indent=2, ensure_ascii=False))
    mp.write_text(json.dumps(index.meta.to_dict(), indent=2, ensure_ascii=False))
    return ip
 
 
def load_index(cwd: str | os.PathLike[str]) -> Index | None:
    ip = index_path(cwd)
    if not ip.is_file():
        return None
    try:
        data = json.loads(ip.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Index.from_dict(data)
 
 
def load_meta(cwd: str | os.PathLike[str]) -> IndexMeta | None:
    mp = meta_path(cwd)
    if not mp.is_file():
        return None
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return IndexMeta.from_dict(data)
 
 
__all__ = [
    "INDEX_DIR",
    "INDEX_FILE",
    "META_FILE",
    "index_dir",
    "index_path",
    "meta_path",
    "save_index",
    "load_index",
    "load_meta",
]
