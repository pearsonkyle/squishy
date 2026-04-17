"""Async agent loop. Safety patterns ported from atlas-proxy/agent.go:19-256.
 
- Conversation trim: system + first_user + last 8 (agent.go:41-50)
- Consecutive read nudge: warn at 4, refuse at 5 (agent.go:203-241)
- Error loop breaker: 3 consecutive tool failures -> stop (agent.go:38)
- write_file-too-big guard: tools/fs.py:_write_file (agent.go:142-167)
- Task-level timeout: asyncio.timeout wraps the entire run()
- Cancellation: asyncio.CancelledError re-raised cleanly
"""
 
from __future__ import annotations
 
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
 
from squishy.client import Client, ToolCall
from squishy.config import Config
from squishy.context import build_system_prompt, detect_project, trim_history
from squishy.display import Display
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.tools import PromptFn, ToolContext, dispatch, openai_schemas
 
log = logging.getLogger("squishy.agent")
 
READ_ONLY_TOOLS = {"read_file", "list_directory", "search_files"}
MAX_READS_BEFORE_WARN = 4
MAX_READS_BEFORE_REFUSE = 5
MAX_CONSECUTIVE_ERRORS = 3
 
 
@dataclass
class TaskResult:
    success: bool
    final_text: str = ""
    turns_used: int = 0
    tokens_used: int = 0
    files_created: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    commands_run: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
 
 
@dataclass
class Agent:
    config: Config
    client: Client
    display: Display | None = None
    prompt_fn: PromptFn | None = None
    tool_ctx: ToolContext = field(init=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
 
    def __post_init__(self) -> None:
        self.tool_ctx = ToolContext(
            working_dir=self.config.working_dir,
            permission_mode=self.config.permission_mode,
            sandbox_image=self.config.sandbox_image,
            use_sandbox=self.config.use_sandbox,
        )
        project = detect_project(self.config.working_dir)
        self.messages.append(
            {
                "role": "system",
                "content": build_system_prompt(
                    self.config.working_dir, project, self.config.thinking
                ),
            }
        )
        self._check_index_staleness()
 
    def _check_index_staleness(self) -> None:
        if self.display is None:
            return
        try:
            from squishy.index.staleness import describe_staleness
        except Exception:  # noqa: BLE001
            return
        msg = describe_staleness(self.config.working_dir)
        if msg:
            self.display.info(msg)
 
    async def run(
        self, user_message: str, *, timeout: float | None = None
    ) -> TaskResult:
        """Run one user turn to completion. Returns a TaskResult.
 
        Raises:
            AgentTimeout: if the task exceeds ``timeout`` seconds.
            AgentCancelled: if the caller cancels the task.
            LLMError: on unrecoverable LLM errors (retries exhausted).
        """
        self.messages.append({"role": "user", "content": user_message})
        start = time.monotonic()
 
        try:
            if timeout is not None:
                async with asyncio.timeout(timeout):
                    return await self._run_loop(start)
            return await self._run_loop(start)
        except TimeoutError as e:
            raise AgentTimeout(f"task exceeded {timeout}s") from e
        except asyncio.CancelledError:
            raise AgentCancelled("task cancelled by caller") from None
 
    async def _run_loop(self, start: float) -> TaskResult:
        consecutive_reads = 0
        consecutive_errors = 0
        schemas = openai_schemas()
        result = TaskResult(success=False)
        tokens = 0
        files_created: set[str] = set()
        files_edited: set[str] = set()
        commands_run = 0
 
        for turn in range(1, self.config.max_turns + 1):
            self.tool_ctx.permission_mode = self.config.permission_mode
            self.messages[:] = trim_history(self.messages)
 
            try:
                completion = await self.client.complete(
                    self.messages,
                    schemas,
                    stream=True,
                    on_text=self._on_text,
                )
            except LLMError as e:
                if self.display:
                    self.display.error(f"LLM error: {e}")
                result.error = str(e)
                result.turns_used = turn - 1
                result.elapsed_s = time.monotonic() - start
                return result
 
            if completion.text and self.display:
                self.display.console.print()
            tokens += completion.usage.get("total_tokens", 0)
 
            if not completion.tool_calls:
                if completion.text:
                    self.messages.append({"role": "assistant", "content": completion.text})
                if self.display:
                    self.display.summary(turn, time.monotonic() - start)
                result.success = True
                result.final_text = completion.text
                result.turns_used = turn
                result.tokens_used = tokens
                result.files_created = sorted(files_created)
                result.files_edited = sorted(files_edited)
                result.commands_run = commands_run
                result.elapsed_s = time.monotonic() - start
                result.messages = list(self.messages)
                return result
 
            self.messages.append(_assistant_msg(completion.text, completion.tool_calls))
 
            any_mutating = False
            for tc in completion.tool_calls:
                if tc.name in READ_ONLY_TOOLS:
                    consecutive_reads += 1
                    if consecutive_reads == MAX_READS_BEFORE_WARN and self.display:
                        self.display.warn("4 consecutive reads — nudging model to write.")
                    if consecutive_reads >= MAX_READS_BEFORE_REFUSE:
                        msg = "You have enough context. Write or edit now. Further reads refused."
                        if self.display:
                            self.display.warn(msg)
                        self._append_tool_result(tc, message=msg)
                        continue
                else:
                    any_mutating = True
 
                outcome = await self._run_tool(turn, tc)
                if outcome["success"]:
                    consecutive_errors = 0
                    if tc.name == "write_file":
                        files_created.add(str(tc.args.get("path", "?")))
                    elif tc.name == "edit_file":
                        files_edited.add(str(tc.args.get("path", "?")))
                    elif tc.name == "run_command":
                        commands_run += 1
                else:
                    consecutive_errors += 1
 
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    msg = f"{MAX_CONSECUTIVE_ERRORS} consecutive tool failures — stopping."
                    if self.display:
                        self.display.error(msg)
                    result.error = msg
                    result.turns_used = turn
                    result.tokens_used = tokens
                    result.files_created = sorted(files_created)
                    result.files_edited = sorted(files_edited)
                    result.commands_run = commands_run
                    result.elapsed_s = time.monotonic() - start
                    result.messages = list(self.messages)
                    return result
 
            if any_mutating:
                consecutive_reads = 0
 
        msg = f"max turns ({self.config.max_turns}) reached"
        if self.display:
            self.display.warn(msg)
            self.display.summary(self.config.max_turns, time.monotonic() - start)
        result.error = msg
        result.turns_used = self.config.max_turns
        result.tokens_used = tokens
        result.files_created = sorted(files_created)
        result.files_edited = sorted(files_edited)
        result.commands_run = commands_run
        result.elapsed_s = time.monotonic() - start
        result.messages = list(self.messages)
        return result
 
    async def _run_tool(self, turn: int, tc: ToolCall) -> dict[str, Any]:
        brief = _brief(tc)
        if self.display:
            self.display.turn_header(turn, self.config.max_turns, tc.name, brief)
 
        if tc.name == "edit_file" and self.display:
            old_str = str(tc.args.get("old_str", ""))
            new_str = str(tc.args.get("new_str", ""))
            if old_str and new_str:
                self.display.edit_diff(str(tc.args.get("path", "")), old_str, new_str)
 
        t0 = time.monotonic()
        outcome = await dispatch(tc.name, tc.args, self.tool_ctx, prompt_fn=self.prompt_fn)
        dt_ms = (time.monotonic() - t0) * 1000
 
        if self.display:
            self.display.tool_result(outcome.success, outcome.display or outcome.error, dt_ms)
            if outcome.success and tc.name == "write_file":
                self.display.write_preview(
                    str(tc.args.get("path", "?")), str(tc.args.get("content", ""))
                )
            if outcome.success:
                if tc.name == "write_file":
                    self.display.stats.files_created.add(str(tc.args.get("path", "?")))
                elif tc.name == "edit_file":
                    self.display.stats.files_edited.add(str(tc.args.get("path", "?")))
                elif tc.name == "run_command":
                    self.display.stats.commands_run += 1
 
        self._append_tool_result(tc, message=outcome.to_message())
        return {"success": outcome.success}
 
    def _append_tool_result(self, tc: ToolCall, message: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": message,
            }
        )
 
    async def _on_text(self, chunk: str) -> None:
        if self.display:
            self.display.streaming_text_chunk(chunk)
 
 
def _assistant_msg(text: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args, ensure_ascii=False)},
            }
            for tc in tool_calls
        ],
    }
 
 
def _brief(tc: ToolCall) -> str:
    a = tc.args
    if tc.name in ("read_file", "write_file", "edit_file", "list_directory"):
        return str(a.get("path", ""))
    if tc.name == "search_files":
        return f'"{a.get("pattern", "")}"'
    if tc.name == "run_command":
        cmd = str(a.get("command", ""))
        return cmd[:60] + ("…" if len(cmd) > 60 else "")
    return ""