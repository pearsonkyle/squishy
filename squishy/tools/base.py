"""Tool data types. Async-native."""
 
from __future__ import annotations
 
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from squishy.plan_state import PlanState

ToolRun = Callable[[dict[str, Any], "ToolContext"], Awaitable["ToolResult"]]
 
 
@dataclass
class ToolContext:
    working_dir: str
    files_read: dict[str, str] = field(default_factory=dict)
    files_read_meta: dict[tuple[str, int, Any], dict[str, Any]] = field(default_factory=dict)
    permission_mode: str = "edits"
    sandbox_image: str = "python:3.11-slim"
    use_sandbox: bool = True
    plan: PlanState | None = None
    pending_plan_evidence: list[dict[str, Any]] = field(default_factory=list)
    plan_switch_prompted: bool = False
    notes: dict[str, str] = field(default_factory=dict)
    # Keys in `notes` seeded by the harness (e.g. FAIL_TO_PASS metadata) that
    # the model's save_note must not evict or overwrite.
    reserved_note_keys: set[str] = field(default_factory=set)
    _cached_index: Any = field(default=None, repr=False)
    _cached_index_mtime: float = field(default=-1.0, repr=False)
    files_read_count: dict[str, int] = field(default_factory=dict)
    edit_fail_files: set[str] = field(default_factory=set)
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
