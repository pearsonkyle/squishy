"""Write-side tool implementations: edit, create, run.

Kept separate from :mod:`graphagent.agentkit.tools`, which is read-only, so
the benchmark can hand an arm exploration tools without also handing it the
ability to mutate the tree. Same rules as the read side: pure functions, no
SDK imports, every path validated against the repository root, and errors
returned as text rather than raised into the agent loop.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from graphagent.agentkit.tools import _safe_resolve

_MAX_OUTPUT_CHARS = 8000
_DEFAULT_TIMEOUT = 300
_SCRATCH = Path(tempfile.gettempdir()).resolve()


def _resolve_writable(root: Path, relative: str) -> Path:
    """Repo paths, plus absolute paths under the system temp directory.

    The agent is told to put reproduction scripts in /tmp so they stay out of
    the graded diff; refusing the very path we asked for is the failure this
    exception exists to avoid. Everywhere else the repo-root guard stands.
    """
    candidate = Path(relative)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved == _SCRATCH or _SCRATCH in resolved.parents:
            return resolved
        raise ValueError(
            f"absolute path outside {_SCRATCH}/ is not writable: {relative!r}"
        )
    return _safe_resolve(root, relative)


def write_file(root: Path, relative: str, content: str) -> str:
    """Create or overwrite a file, creating parent directories as needed."""
    try:
        target = _resolve_writable(root, relative)
    except ValueError as exc:
        return f"error: {exc}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(content)
    return f"wrote {relative} ({len(content.splitlines())} lines)"


def edit_file(root: Path, relative: str, old_str: str, new_str: str) -> str:
    """Replace one unique occurrence of ``old_str`` with ``new_str``.

    A miss returns the reason plus what to do about it. The failure mode this
    guards against is a model that retries the identical non-matching string
    until its turn budget runs out: telling it *how many* times the string
    occurred distinguishes "wrong text" from "not unique enough", which are
    opposite fixes.
    """
    try:
        target = _resolve_writable(root, relative)
    except ValueError as exc:
        return f"error: {exc}"
    if not target.is_file():
        return f"error: no such file: {relative} (use write_file to create it)"
    with target.open(encoding="utf-8") as handle:
        text = handle.read()
    count = text.count(old_str)
    if count == 0:
        return (
            f"error: old_str not found in {relative}. Read the exact lines "
            "first (read_file or symbol_source) and copy them verbatim, "
            "including indentation."
        )
    if count > 1:
        return (
            f"error: old_str appears {count} times in {relative}; include "
            "surrounding lines so the match is unique."
        )
    with target.open("w", encoding="utf-8") as handle:
        handle.write(text.replace(old_str, new_str, 1))
    line = text[: text.index(old_str)].count("\n") + 1
    return f"edited {relative} at line {line}"


def run_command(root: Path, command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Run a shell command in the repository root and return its output."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s: {command}"
    output = (proc.stdout + proc.stderr).strip()
    if len(output) > _MAX_OUTPUT_CHARS:
        half = _MAX_OUTPUT_CHARS // 2
        output = f"{output[:half]}\n… (output truncated) …\n{output[-half:]}"
    return f"exit={proc.returncode}\n{output}" if output else f"exit={proc.returncode}"
