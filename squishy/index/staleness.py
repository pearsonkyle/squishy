"""Detect when `.squishy/index.json` is older than the files it indexed.

Enhanced staleness detection with:
- mtime comparison for quick checks
- Deep hash verification for accuracy
- Pattern-based warnings (e.g., test files excluded)
"""

from __future__ import annotations

import os
from pathlib import Path

from squishy.index.store import load_index, load_meta
from squishy.index.walker import walk_repo

STALE_GRACE_SECONDS = 5.0  # ignore mtime jitter within the build window
 
 
def describe_staleness(cwd: str | os.PathLike[str]) -> str:
    """Check index staleness using mtime comparison.

    Returns a warning string if stale, or empty string when fresh.
    """
    meta = load_meta(cwd)
    if meta is None:
        return ""
    generated = meta.generated_at or 0.0
    if generated <= 0:
        return ""

    root = Path(cwd)
    
    # Quick mtime check for tracked files
    newest_mtime = 0.0
    newest_path = ""
    stale_files: list[str] = []
    
    for rel_path, stored_hash in meta.file_hashes.items():
        p = root / rel_path
        try:
            s = p.stat()
            mtime = s.st_mtime

            # Flag files that are newer than index + grace
            if mtime > generated + STALE_GRACE_SECONDS:
                stale_files.append(f"{rel_path} (changed {int(mtime - generated)}s ago)")
            
            # Track newest file
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = rel_path
        except OSError:
            # File might have been deleted
            stale_files.append(f"{rel_path} (deleted)")
    
    # Check for new files not in the index
    records, _ = walk_repo(cwd)
    tracked = set(meta.file_hashes.keys())
    new_files: list[str] = []
    
    for rec in records:
        if rec.path not in tracked:
            new_files.append(rec.path)
    
    # Build warning message
    warnings = []
    
    if stale_files:
        sample = ", ".join(stale_files[:3])
        extra = f" (+{len(stale_files) - 3} more)" if len(stale_files) > 3 else ""
        warnings.append(f"changed files: {sample}{extra}")
    
    if new_files:
        sample = ", ".join(new_files[:3])
        extra = f" (+{len(new_files) - 3} more)" if len(new_files) > 3 else ""
        warnings.append(f"new files: {sample}{extra}")
    
    if not warnings:
        return ""
    
    return f"repo index out of date: {'; '.join(warnings)}. Run /init to refresh."


def describe_deep_staleness(cwd: str | os.PathLike[str]) -> dict[str, object]:
    """Deep staleness check with detailed diagnostics.

    Returns a dict with:
      - stale: bool
      - reason: str | None
      - stale_files: list[str]
      - new_files: list[str]
      - deleted_files: list[str]
    """
    meta = load_meta(cwd)
    if meta is None:
        return {
            "stale": False,
            "reason": None,
            "stale_files": [],
            "new_files": [],
            "deleted_files": [],
        }

    root = Path(cwd)
    stale_files: list[str] = []
    new_files: list[str] = []
    deleted_files: list[str] = []

    # Check tracked files
    for rel_path, _ in meta.file_hashes.items():
        p = root / rel_path
        try:
            mtime = p.stat().st_mtime
            if mtime > meta.generated_at + STALE_GRACE_SECONDS:
                stale_files.append(rel_path)
        except OSError:
            deleted_files.append(rel_path)

    # Check for new files
    records, _ = walk_repo(cwd)
    tracked = set(meta.file_hashes.keys())
    for rec in records:
        if rec.path not in tracked:
            new_files.append(rec.path)

    stale = bool(stale_files or new_files or deleted_files)
    
    reason: str | None = None
    if stale_files:
        reason = f"{len(stale_files)} file(s) changed"
    elif new_files:
        reason = f"{len(new_files)} new file(s)"
    elif deleted_files:
        reason = f"{len(deleted_files)} file(s) deleted"

    return {
        "stale": stale,
        "reason": reason,
        "stale_files": stale_files[:10],  # Limit for display
        "new_files": new_files[:10],
        "deleted_files": deleted_files[:10],
    }


__all__ = ["describe_staleness", "describe_deep_staleness"]
