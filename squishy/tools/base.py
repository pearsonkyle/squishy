"""Tool data types. Async-native."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from squishy.config import Config
    from squishy.display import Display
    from squishy.plan_tracker import PlanTracker

ToolRun = Callable[[dict[str, Any], "ToolContext"], Awaitable["ToolResult"]]
# Prompt function used by tools that need a free-form yes/no confirmation
# (e.g. exit_plan_mode). Distinct from the per-tool permission prompt in
# tools/__init__.py, which fires before dispatch.
ApproveFn = Callable[[str], Awaitable[bool]]


@dataclass
class ToolContext:
    working_dir: str
    files_read: dict[str, str] = field(default_factory=dict)
    permission_mode: str = "edits"
    sandbox_image: str = "python:3.11-slim"
    use_sandbox: bool = True
    # Optional shared state for plan-mode tools. These are set when the Agent
    # owns a Config/Display/PlanTracker (i.e. always in CLI use, sometimes in
    # tests). Plan tools fail gracefully when unset.
    cfg_ref: "Config | None" = None
    tracker: "PlanTracker | None" = None
    display: "Display | None" = None
    approve_fn: ApproveFn | None = None
 
 
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
    mutates: bool = False
 
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