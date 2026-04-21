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
from dataclasses import dataclass, field, replace
from typing import Any
 
from squishy.agent import Agent, TaskResult
from squishy.client import Client
from squishy.config import MODES, Config, PermissionMode
 
 
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
    ) -> TaskResult:
        """Run a single user turn to completion.
 
        Args:
            message: user instruction
            working_dir: directory the agent operates in (defaults to cwd)
            timeout: overall task timeout in seconds; raises AgentTimeout if exceeded
            on_text: optional callback for streaming text chunks from the model
        """
        cfg = self._make_config(working_dir)
        display = _CallbackDisplay(on_text) if on_text else None
        agent = Agent(cfg, self._client, display=display)  # type: ignore[arg-type]
        return await agent.run(message, timeout=timeout)
 
    def _make_config(self, working_dir: str | None) -> Config:
        cfg = Config()
        return replace(
            cfg,
            permission_mode=self.permission_mode,
            max_turns=self.max_turns,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            working_dir=working_dir or cfg.working_dir,
            sandbox_image=self.sandbox_image,
            use_sandbox=self.use_sandbox,
            thinking=self.thinking,
        )
 
 
class _CallbackDisplay:
    """Display adapter that forwards streamed text to a user callback.
 
    Only implements the subset of Display the Agent actually uses when the
    programmatic API is driving. No rich output — keeps library usage silent
    unless the caller opts into streaming.
    """
 
    def __init__(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text
        self.stats = _Stats()
 
    class _Console:
        def print(self, *_: Any, **__: Any) -> None:
            pass
 
        def out(self, *_: Any, **__: Any) -> None:
            pass
 
    console = _Console()
 
    def streaming_text_chunk(self, chunk: str) -> None:
        with contextlib.suppress(Exception):
            self._on_text(chunk)
 
    def turn_header(self, *_: Any, **__: Any) -> None: ...
    def tool_result(self, *_: Any, **__: Any) -> None: ...
    def edit_diff(self, *_: Any, **__: Any) -> None: ...
    def write_preview(self, *_: Any, **__: Any) -> None: ...
    def command_output(self, *_: Any, **__: Any) -> None: ...
    def text(self, *_: Any, **__: Any) -> None: ...
    def info(self, *_: Any, **__: Any) -> None: ...
    def warn(self, *_: Any, **__: Any) -> None: ...
    def error(self, *_: Any, **__: Any) -> None: ...
    def plan_panel(self, *_: Any, **__: Any) -> None: ...
    def plan_progress(self, *_: Any, **__: Any) -> None: ...
    def summary(self, *_: Any, **__: Any) -> None: ...
 
 
@dataclass
class _Stats:
    files_created: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    commands_run: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_window: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


__all__ = ["Squishy", "TaskResult"]
