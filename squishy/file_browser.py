"""File browser: parse @filename references and inject file contents.

Users can reference files in their input using @filename syntax.
This module extracts those references, reads the file contents, and
wraps them in a delimiter that the LLM recognizes as file content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from squishy.tools.fs import _resolve

# Match @path references, but NOT the @ inside an email address or a path
# (negative lookbehind for a word char / @ / /), and capture the run of
# non-space chars. Trailing sentence punctuation is stripped afterwards so
# `@app.py.` / `@app.py,` resolve to `app.py`.
FILE_PATTERN = re.compile(r"(?<![\w@/])@(\S+)")
"""Regex pattern to match @filename references."""

# Punctuation that is almost always sentence/markup, not part of a filename.
_TRAILING_PUNCT = ".,;:!?\"')]}>"


def _clean_ref(raw: str) -> str:
    """Strip trailing sentence punctuation from a captured @reference."""
    return raw.rstrip(_TRAILING_PUNCT)


FILE_WRAPPER = """<file path="{path}" total_lines="{total_lines}">{content}
</file>"""
"""Wrapper format for file contents sent to the LLM."""


@dataclass
class FileReference:
    """Represents a file reference from user input."""

    path: str  # Original path as written by user
    absolute_path: str  # Resolved absolute path
    content: str  # File contents


def parse_references(text: str, working_dir: str) -> list[FileReference]:
    """Find all @filename references in text and return their contents.

    Files that can't be read are silently skipped — use
    ``parse_references_with_missing`` when the caller wants to warn
    the user about typos like ``@nonexistant.py``.
    """
    found, _ = parse_references_with_missing(text, working_dir)
    return found


def parse_references_with_missing(
    text: str, working_dir: str,
) -> tuple[list[FileReference], list[str]]:
    """Like ``parse_references`` but also returns the @paths that
    could not be read, so the caller can warn the user.

    A @reference that fails to resolve is otherwise silently ignored —
    the literal ``@typo.py`` ends up in the prompt sent to the model
    with no feedback to the user that their reference was a no-op.
    """
    references: list[FileReference] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in FILE_PATTERN.findall(text):
        path = _clean_ref(raw)
        if not path or path in seen:
            continue
        seen.add(path)
        abs_path = _resolve(path, working_dir)
        content = _read_file(abs_path)
        if content is not None:
            references.append(FileReference(
                path=path,
                absolute_path=abs_path,
                content=content,
            ))
        else:
            missing.append(path)
    return references, missing


def _read_file(abs_path: str) -> str | None:
    """Read a file and return its contents, or None if not readable."""
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def inject_references(text: str, working_dir: str) -> tuple[str, list[FileReference]]:
    """Replace @filename references with wrapped file contents.

    Args:
        text: User input potentially containing @filename patterns
        working_dir: Working directory for resolving relative paths

    Returns:
        Tuple of (modified text with wrapped contents, list of references).
        Missing references are not surfaced here for backwards
        compatibility — callers wanting to warn on typos should use
        ``inject_references_with_missing``.
    """
    result, references, _ = inject_references_with_missing(text, working_dir)
    return result, references


def inject_references_with_missing(
    text: str, working_dir: str,
) -> tuple[str, list[FileReference], list[str]]:
    """Like ``inject_references`` but also returns the @paths that could
    not be read.

    Missing entries are stripped from the outgoing message text — leaving
    a literal ``@typo.py`` confuses weak models — and replaced with an
    inline marker so the model knows the user *meant* to attach a file.
    """
    references, missing = parse_references_with_missing(text, working_dir)

    result = text
    for ref in references:
        wrapped = FILE_WRAPPER.format(
            path=ref.path,
            total_lines=len(ref.content.splitlines()),
            content=ref.content,
        )
        result = result.replace(f"@{ref.path}", wrapped, 1)

    for missing_path in missing:
        result = result.replace(
            f"@{missing_path}",
            f"[file not found: {missing_path}]",
            1,
        )

    return result, references, missing


def format_reference_list(references: list[FileReference]) -> str:
    """Format a list of file references for display purposes.

    Args:
        references: List of FileReference objects

    Returns:
        Human-readable summary of referenced files
    """
    if not references:
        return ""

    lines = ["Referenced files:"]
    for ref in references:
        line_count = len(ref.content.splitlines())
        lines.append(f"  - {ref.path} ({line_count} lines)")

    return "\n".join(lines)
