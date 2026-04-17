"""Detect when `.squishy/index.json` is older than the files it indexed.
 
Cheap mtime comparison — no hashing, no tree walk. Returns a short nudge
string to print at Agent startup, or `""` when fresh / no index exists.
"""
 
from __future__ import annotations
 
import os
from pathlib import Path
 
from squishy.index.store import load_meta
from squishy.index.walker import walk_repo
 
STALE_GRACE_SECONDS = 5.0  # ignore mtime jitter within the build window
 
 
def describe_staleness(cwd: str | os.PathLike[str]) -> str:
    meta = load_meta(cwd)
    if meta is None:
        return ""
    generated = meta.generated_at or 0.0
    if generated <= 0:
        return ""
 
    # Look only at tracked file mtimes, not the whole tree. Bail on the first
    # file that's newer than (generated_at + grace).
    root = Path(cwd)
    newest = 0.0
    newest_path = ""
    for rel in meta.file_hashes:
        p = root / rel
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > newest:
            newest = m
            newest_path = rel
        if m > generated + STALE_GRACE_SECONDS:
            break
    # Also flag new files that weren't in the prior hash set.
    if newest <= generated + STALE_GRACE_SECONDS:
        records, _ = walk_repo(cwd)
        tracked = set(meta.file_hashes)
        new_files = [r.path for r in records if r.path not in tracked]
        if new_files:
            sample = ", ".join(new_files[:3])
            extra = f" (+{len(new_files) - 3} more)" if len(new_files) > 3 else ""
            return f"repo index out of date: {len(new_files)} new file(s): {sample}{extra}. Run /init to refresh."
        return ""
    return f"repo index out of date: {newest_path} changed after last /init. Run /init to refresh."
 
 
__all__ = ["describe_staleness"]
