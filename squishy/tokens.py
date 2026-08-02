"""Canonical token-count estimation (no tokenizer dependency).

Single home for the ~3.5 chars/token heuristic that was previously copied
into ``display.py`` and ``context.py`` (three call sites, one of which
re-walked the message list). All estimates route through here so the constant
and the per-message overhead live in exactly one place.

3.5 chars/token is a deliberately code-friendly ratio (the common 4 chars/token
under-counts dense source); ``PER_MSG_OVERHEAD`` accounts for the role/format
tokens the chat API adds around each message.
"""

from __future__ import annotations

import math
from typing import Any

CHARS_PER_TOKEN = 3.5
PER_MSG_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """Estimate tokens for one text blob (system prompt, streamed chunk, ...).

    Rounds up so a short non-empty string still costs at least one token plus
    overhead; an empty string costs nothing.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN) + PER_MSG_OVERHEAD


def message_chars(m: dict[str, Any]) -> int:
    """Character weight of one chat message (content + tool-call payloads)."""
    chars = 0
    content = m.get("content", "")
    if isinstance(content, str):
        chars += len(content)
    for tc in m.get("tool_calls", []) or []:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            chars += len(func.get("name", "")) + len(func.get("arguments", ""))
    return chars


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate tokens for a full message list (compaction / trim sizing).

    Uses truncation (not ceil) per message to match the long-standing
    compaction-threshold behavior exactly.
    """
    return sum(
        int(message_chars(m) / CHARS_PER_TOKEN) + PER_MSG_OVERHEAD
        for m in messages
    )


__all__ = [
    "CHARS_PER_TOKEN",
    "PER_MSG_OVERHEAD",
    "estimate_tokens",
    "message_chars",
    "estimate_message_tokens",
]
