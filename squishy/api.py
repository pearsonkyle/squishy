"""High-level programmatic API.

Use from Python code or benchmark harnesses:

    async with Squishy(model="local-model") as sq:
        result = await sq.run("fix the bug in app.py", working_dir="/tmp/repo", timeout=300)
        print(result.final_text, result.files_edited)

`Squishy` is the stable public surface. Prefer it over constructing Agent/Client directly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from typing import Any

from squishy.agent import Agent, TaskResult
from squishy.client import Client
from squishy.config import MODES, Config, PermissionMode
from squishy.display import Stats

# Field names shared between Squishy and Config (for _make_config).
_CONFIG_FIELDS = frozenset(
    f.name for f in fields(Config) if f.name != "working_dir"
)


@dataclass
class Squishy:
    """Facade around Config + Client + Agent, suitable for library use and benchmarks."""

    model: str
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local"
    temperature: float = 0.3
    max_tokens: int = 8192
    max_turns: int = 20
    permission_mode: PermissionMode = "yolo"
    request_timeout: float = 120.0
    max_retries: int = 4
    use_sandbox: bool = False
    sandbox_image: str = "python:3.11-slim"
    thinking: bool = False
    max_consecutive_errors: int = 3
    max_plan_nudges: int = 4
    max_plan_investigation_turns: int = 4
    max_recall_skip_turns: int = 2
    max_history_messages: int = 10
    max_quality_retries: int = 2
    compaction_threshold: float = 0.7
    max_stuck_turns: int = 3
    max_explore_turns: int = 8
    max_fix_verify_cycles: int = 6
    max_post_edit_read_turns: int = 4
    max_tool_output_chars: int = 32_000
    auto_init: bool = False

    _client: Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.permission_mode not in MODES:
            raise ValueError(f"permission_mode must be one of {MODES}")
        self._client = Client(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

    async def __aenter__(self) -> Squishy:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        return await self._client.health()

    async def run(
        self,
        message: str,
        *,
        working_dir: str | None = None,
        timeout: float | None = None,
        on_text: Callable[[str], None] | None = None,
        session_id: str | None = None,
    ) -> TaskResult:
        """Run a single user turn to completion."""
        cfg = self._make_config(working_dir)
        display = _CallbackDisplay(on_text) if on_text else None
        agent = Agent(cfg, self._client, display=display, session_id=session_id)  # type: ignore[arg-type]
        return await agent.run(message, timeout=timeout)

    def _make_config(self, working_dir: str | None) -> Config:
        overrides = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name in _CONFIG_FIELDS and not f.name.startswith("_")
        }
        if working_dir:
            overrides["working_dir"] = working_dir
        return replace(Config(), **overrides)


class _CallbackDisplay:
    """Display adapter that forwards streamed text to a user callback.

    Uses __getattr__ fallback for all Display methods the Agent calls
    that we don't need to handle (turn_header, tool_result, etc.).
    """

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text
        self.stats = Stats()
        self.console = type(
            "_NullConsole", (),
            {"print": lambda *a, **k: None, "out": lambda *a, **k: None},
        )()

    def streaming_text_chunk(self, chunk: str) -> None:
        with contextlib.suppress(Exception):
            self._on_text(chunk)

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **kw: None


__all__ = ["Squishy", "TaskResult"]
