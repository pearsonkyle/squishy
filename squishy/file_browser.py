"""File browser: parse @filename references and inject file contents.

Users can reference files in their input using @filename syntax.
This module extracts those references, reads the file contents, and
wraps them in a delimiter that the LLM recognizes as file content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from squishy.tools.fs import _resolve

FILE_PATTERN = re.compile(r"@(\S+)")
"""Regex pattern to match @filename references."""


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
    """Find all @filename references in text and return their contents."""
    references = []
    matches = FILE_PATTERN.findall(text)

    for path in matches:
        abs_path = _resolve(path, working_dir)
        content = _read_file(abs_path)
        if content is not None:
            references.append(FileReference(
                path=path,
                absolute_path=abs_path,
                content=content,
            ))

    return references


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
        Tuple of (modified text with wrapped contents, list of references)
    """
    references = parse_references(text, working_dir)

    # Replace each @filename with its wrapped content
    result = text
    for ref in references:
        # Wrap the content with file metadata
        wrapped = FILE_WRAPPER.format(
            path=ref.path,
            total_lines=len(ref.content.splitlines()),
            content=ref.content,
        )
        # Replace @filename with wrapped content
        result = result.replace(f"@{ref.path}", wrapped, 1)

    return result, references


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
