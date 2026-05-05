"""Walk a repository and yield source files.

Honors ``SKIP_DIRS`` from ``squishy.tools.fs``, every ``.gitignore``
in the tree (via ``squishy.index.gitignore``), and a hard cap to keep
runaway monorepos from DoSing ``/init``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from squishy.index.gitignore import GitignoreFilter, discover_gitignores
from squishy.tools.fs import SKIP_DIRS
 
FILE_CAP = 5000
FILE_WARN = 2000
MAX_BYTES = 512 * 1024  # skip files > 512KB; they're almost never worth indexing
 
# Extensions we know how to index. Everything else is recorded as a file node
# without symbols (so docs/config still show up in the tree and in `recall`).
SOURCE_EXTS = {
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".go",
    ".rs",
    ".c", ".h", ".cc", ".hh", ".cpp", ".hpp",
    ".java",
    ".rb",
    ".php",
    ".swift",
    ".kt", ".kts",
    ".lua",
    ".sh", ".bash",
}
TEXT_EXTS = SOURCE_EXTS | {
    ".md", ".rst", ".txt",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
    ".html", ".css", ".scss",
}
 
 
@dataclass
class FileRecord:
    path: str  # posix-style, relative to root
    abs_path: str
    size: int
    ext: str
    hash: str  # blake2 of contents
 
 
def load_gitignore(root: Path) -> GitignoreFilter:
    """Load every ``.gitignore`` under ``root`` into one matcher.

    Public so AGENTS.md generation and other tree consumers can apply
    the same filter without duplicating logic.
    """
    return discover_gitignores(root, skip_dirs=SKIP_DIRS)
 
 
def _blake2(abs_path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    try:
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
 
 
def walk_repo(cwd: str | os.PathLike[str]) -> tuple[list[FileRecord], bool]:
    """Walk `cwd` and return `(records, hit_cap)`.
 
    `hit_cap` is True when we stopped collecting at `FILE_CAP` — callers can
    warn the user that the index is partial.
    """
    root = Path(cwd).resolve()
    ignore = load_gitignore(root)
    records: list[FileRecord] = []
    hit_cap = False

    for dirpath, dirs, files in os.walk(root):
        # Filter directories: SKIP_DIRS, dotfiles, and gitignored dirs.
        # Pruning here means we never recurse into them, which keeps the
        # walk fast on large repos.
        try:
            cur_rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        except ValueError:
            cur_rel = ""
        if cur_rel == ".":
            cur_rel = ""

        kept: list[str] = []
        for d in sorted(dirs):
            if d in SKIP_DIRS or d.startswith("."):
                continue
            sub_rel = f"{cur_rel}/{d}" if cur_rel else d
            if ignore.is_ignored(sub_rel, is_dir=True):
                continue
            kept.append(d)
        dirs[:] = kept

        for name in sorted(files):
            if name.startswith("."):
                continue
            abs_path = os.path.join(dirpath, name)
            try:
                rel = os.path.relpath(abs_path, root)
            except ValueError:
                continue
            rel_posix = rel.replace(os.sep, "/")
            if ignore.is_ignored(rel_posix):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext and ext not in TEXT_EXTS:
                continue
            try:
                size = os.path.getsize(abs_path)
            except OSError:
                continue
            if size > MAX_BYTES:
                continue
            h = _blake2(abs_path)
            records.append(FileRecord(
                path=rel_posix,
                abs_path=abs_path,
                size=size,
                ext=ext,
                hash=h,
            ))
            if len(records) >= FILE_CAP:
                hit_cap = True
                return records, hit_cap
    return records, hit_cap
 
 
__all__ = [
    "FileRecord",
    "FILE_CAP",
    "FILE_WARN",
    "MAX_BYTES",
    "SOURCE_EXTS",
    "TEXT_EXTS",
    "walk_repo",
]
