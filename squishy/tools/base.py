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
 
 
@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    display: str = ""
 
    def to_message(self) -> str:
        if self.success:
            return _short_json(self.data)
        return _short_json({"error": self.error})
 
 
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
 
 
def _short_json(d: dict[str, Any], limit: int = 4000) -> str:
    import json
 
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "...<truncated>"
