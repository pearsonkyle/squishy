"""Tool data types. Async-native."""
 
from __future__ import annotations
 
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

ToolRun = Callable[[dict[str, Any], "ToolContext"], Awaitable["ToolResult"]]
 
 
@dataclass
class ToolContext:
    working_dir: str
    files_read: dict[str, str] = field(default_factory=dict)
    files_read_meta: dict[tuple[str, int, Any], dict[str, Any]] = field(default_factory=dict)
    permission_mode: str = "edits"
    sandbox_image: str = "python:3.11-slim"
    use_sandbox: bool = True
    notes: dict[str, str] = field(default_factory=dict)
    # Keys in `notes` seeded by the harness (e.g. FAIL_TO_PASS metadata) that
    # the model's save_note must not evict or overwrite.
    reserved_note_keys: set[str] = field(default_factory=set)
    _cached_index: Any = field(default=None, repr=False)
    _cached_index_mtime: float = field(default=-1.0, repr=False)
    _cached_graph: Any = field(default=None, repr=False)
    _cached_graph_mtime: float = field(default=-1.0, repr=False)
    files_read_count: dict[str, int] = field(default_factory=dict)
    # Repeat count per (abs_path, offset, limit) for reads served from cache.
    # Lets read_file escalate from "here it is again" to a hard refusal when a
    # model loops on the identical read.
    read_cache_hits: dict[tuple[str, int, Any], int] = field(default_factory=dict)
    # Line spans already served per path, as (start, end). Lets read_file tell
    # paging through a big file (new ground each time) apart from circling back
    # over content the model already has — only the latter is a loop.
    files_read_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # How many times an equivalent shell command has already returned the same
    # output. Under a shell-only profile this is the only loop-breaking signal
    # available — re-running a grep always succeeds, so nothing else pushes
    # back on a model circling the same three commands.
    command_echoes: dict[str, int] = field(default_factory=dict)
    edit_fail_files: set[str] = field(default_factory=set)
    # Edit pressure, measured against the turn budget rather than guessed.
    # `turns_used`/`turn_budget` are published by the loop each turn so a tool
    # result can say "turn 38 of 50 and nothing edited yet" — the one thing
    # only the loop knows and only the tool result can safely deliver.
    turns_used: int = 0
    turn_budget: int = 0
    # True once a non-scratch file in the repo has actually changed.
    source_edited: bool = False
    # Commands run since the last source edit. A run of them with nothing
    # edited is an investigation that is not converging.
    probe_commands: int = 0
    extra_env: dict[str, str] = field(default_factory=dict)
    # (abs_path, original_content); original_content is None when the entry
    # records a newly-created file (undo deletes it).
    undo_stack: list[tuple[str, str | None]] = field(default_factory=list)
    max_tool_output_chars: int = 32_000
 
 
@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    display: str = ""
 
    def to_message(self, limit: int = 32_000) -> str:
        if self.success:
            return _short_json(self.data, limit)
        return _short_json({"error": self.error}, limit)
 
 
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: ToolRun

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
 
 
def _short_json(d: dict[str, Any], limit: int = 32000) -> str:
    import json
 
    s = json.dumps(d, ensure_ascii=False)
    # limit <= 0 must not grow the payload: s[-0:] is the whole string, so the
    # head+tail path below would return the full content with a bogus banner.
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    head = int(limit * 0.6)
    tail = max(1, int(limit * 0.3))
    snipped = len(s) - head - tail
    return f"{s[:head]}\n[... {snipped} chars snipped ...]\n{s[-tail:]}"
