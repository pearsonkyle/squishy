from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from squishy.client import CompletionResult
from squishy.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)


@dataclass
class FakeClient:
    """Scripted fake LLM client for agent loop tests.

    Pops ``CompletionResult`` objects from ``script`` in order.  When the
    script is exhausted it returns a bare ``CompletionResult(text="done.",
    tool_calls=[])`` so tests don't have to pad the script.
    """

    script: list[CompletionResult]
    calls_seen: list[list[dict[str, Any]]] = field(default_factory=list)
    # Schemas offered per request — lets tests assert on what the model was
    # actually shown, not just what it was asked.
    tools_seen: list[list[dict[str, Any]]] = field(default_factory=list)
    _i: int = 0

    async def health(self) -> bool:
        return True

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: Any = None,
        on_retry: Any = None,
    ) -> CompletionResult:
        self.calls_seen.append(list(messages))
        self.tools_seen.append(list(tools or []))
        if self._i >= len(self.script):
            return CompletionResult(text="done.", tool_calls=[])
        result = self.script[self._i]
        self._i += 1
        return result
