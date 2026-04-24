"""Walk a repository and yield source files.
 
Honors `SKIP_DIRS` from `squishy.tools.fs`, a top-level `.gitignore` (stdlib
`fnmatch`), and a hard cap to keep runaway monorepos from DoSing `/init`.
"""
 
from __future__ import annotations
 
import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
 
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
 
 
def _load_gitignore(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    patterns: list[str] = []
    for raw in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip leading "/" — fnmatch treats patterns as whole-path matches.
        patterns.append(line.lstrip("/"))
    return patterns
 
 
def _gitignored(rel_posix: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    # Match against full relative path and each path segment.
    parts = rel_posix.split("/")
    for pat in patterns:
        # fnmatch doesn't support **; strip leading **/ and match the
        # remainder against the full path and each segment.
        stripped = pat
        while stripped.startswith("**/"):
            stripped = stripped[3:]
        candidates = [pat, stripped] if stripped != pat else [pat]
        for p in candidates:
            if fnmatch.fnmatch(rel_posix, p):
                return True
            if any(fnmatch.fnmatch(seg, p) for seg in parts):
                return True
    return False
 
 
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
    patterns = _load_gitignore(root)
    records: list[FileRecord] = []
    hit_cap = False
 
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        )
        for name in sorted(files):
            if name.startswith("."):
                continue
            abs_path = os.path.join(dirpath, name)
            try:
                rel = os.path.relpath(abs_path, root)
            except ValueError:
                continue
            rel_posix = rel.replace(os.sep, "/")
            if _gitignored(rel_posix, patterns):
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
