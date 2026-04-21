"""Async agent loop. Safety patterns ported from atlas-proxy/agent.go:19-256.

- Conversation trim: system + first_user + last 8 (agent.go:41-50)
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
from squishy.display import Display, estimate_tokens
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.tools import PromptFn, ToolContext, ToolResult, dispatch, openai_schemas

log = logging.getLogger("squishy.agent")

MAX_CONSECUTIVE_ERRORS = 3
# Maximum nudge attempts before giving up on planning
MAX_PLAN_NUDGES = 4
# In plan mode: after this many consecutive tool-call turns without a plan_task
# call, inject a nudge. Reduced from 8 to encourage faster planning.
MAX_PLAN_TOOL_TURNS = 4
# In plan mode: after this many consecutive reads WITHOUT using recall,
# warn the model. Set to 2 to enforce early recall usage.
MAX_RECALL_SKIP_TURNS = 2
 
 
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
    # Track consecutive reads without recall across turns in plan mode
    consecutive_reads_without_recall: int = 0

    def __post_init__(self) -> None:
        self.tool_ctx = ToolContext(
            working_dir=self.config.working_dir,
            permission_mode=self.config.permission_mode,
            sandbox_image=self.config.sandbox_image,
            use_sandbox=self.config.use_sandbox,
        )
        project = detect_project(self.config.working_dir)
        system_prompt = build_system_prompt(
            self.config.working_dir,
            project,
            self.config.thinking,
            self.config.permission_mode,
        )
        self.messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
        
        # Add token estimation for system prompt
        if self.display is not None:
            self.display.stats.prompt_tokens += estimate_tokens(system_prompt)
        
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
        # Add user message (tokens will be counted after trim_history)
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

    def _add_tool_result_messages(self) -> None:
        """Add token estimation for any pending tool result messages in history."""
        if self.display is None:
            return

    def _count_prompt_tokens_from_messages(self) -> int:
        """Estimate prompt tokens from current messages (after trim_history)."""
        total = 0
        for msg in self.messages:
            if msg.get("content"):
                total += estimate_tokens(msg["content"])
        return total

    async def _run_loop(self, start: float) -> TaskResult:
        consecutive_errors = 0
        plan_nudges = 0
        turns_without_plan_task = 0  # plan-mode: read-only turns without plan_task call
        result = TaskResult(success=False)
        total_prompt_tokens = 0
        completion_tokens = 0
        files_created: set[str] = set()
        files_edited: set[str] = set()
        commands_run = 0

        for turn in range(1, self.config.max_turns + 1):
            self.tool_ctx.permission_mode = self.config.permission_mode
            # Schemas are mode-scoped so plan mode never exposes write/edit tools.
            schemas = openai_schemas(self.config.permission_mode)
            # Trim history before sending to LLM
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

            # Count tokens from messages after trimming
            msg_prompt_tokens = self._count_prompt_tokens_from_messages()

            # Track LLM response tokens (total prompt = messages + LLM's reported usage)
            total_prompt_tokens += msg_prompt_tokens
            completion_tokens += completion.completion_tokens

            if not completion.tool_calls:
                in_plan = self.config.permission_mode == "plan"
                plan = getattr(self.tool_ctx, "active_plan", None)
                if in_plan and plan is None:
                    # In plan mode, prose-only completion is not acceptable —
                    # the agent must produce a plan_task call. Nudge up to
                    # MAX_PLAN_NUDGES times, then give up.
                    if completion.text:
                        self.messages.append(
                            {"role": "assistant", "content": completion.text}
                        )
                    if plan_nudges >= MAX_PLAN_NUDGES:
                        msg = "plan-mode run finished without producing a plan_task"
                        if self.display:
                            self.display.error(msg)
                        result.error = msg
                        result.turns_used = turn
                        result.tokens_used = total_prompt_tokens + completion_tokens
                        result.elapsed_s = time.monotonic() - start
                        result.messages = list(self.messages)
                        return result
                    plan_nudges += 1
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[system] You are in plan mode. Stop explaining and call "
                                "`plan_task` now with problem, solution, and steps. "
                                "Use your best current understanding instead of waiting "
                                "for exhaustive research. `files_to_modify` and "
                                "`files_to_create` may be partial or empty if uncertain. "
                                "Do not respond with prose until the plan is approved."
                            ),
                        }
                    )
                    continue
                if completion.text:
                    self.messages.append({"role": "assistant", "content": completion.text})
                if self.display:
                    self.display.stats.prompt_tokens = total_prompt_tokens
                    self.display.stats.completion_tokens = completion_tokens
                    self.display.summary(turn, time.monotonic() - start)
                result.success = True
                result.final_text = completion.text
                result.turns_used = turn
                result.tokens_used = total_prompt_tokens + completion_tokens
                result.files_created = sorted(files_created)
                result.files_edited = sorted(files_edited)
                result.commands_run = commands_run
                result.elapsed_s = time.monotonic() - start
                result.messages = list(self.messages)
                return result

            self.messages.append(_assistant_msg(completion.text, completion.tool_calls))

            plan_task_called_this_turn = False
            # Track consecutive read tools without recall within this turn
            local_read_without_recall = 0
            for tc in completion.tool_calls:
                if tc.name == "plan_task":
                    plan_task_called_this_turn = True
                outcome = await self._run_tool(turn, tc)
                if outcome["success"]:
                    consecutive_errors = 0
                    # Track read tools (read_file, list_directory, search_files)
                    if tc.name in ("read_file", "list_directory", "search_files"):
                        local_read_without_recall += 1
                    elif tc.name == "recall":
                        # Reset the budget when recall is used
                        local_read_without_recall = 0
                        self.consecutive_reads_without_recall = 0  # Also reset persistent counter
                    if tc.name == "write_file":
                        files_created.add(str(tc.args.get("path", "?")))
                    elif tc.name == "edit_file":
                        files_edited.add(str(tc.args.get("path", "?")))
                    elif tc.name == "run_command":
                        commands_run += 1
                else:
                    consecutive_errors += 1

                # In plan mode, an approved plan is the terminal event for the run.
                if (
                    outcome.get("plan_approved")
                    and self.config.permission_mode == "plan"
                ):
                    if self.display:
                        self.display.stats.prompt_tokens = total_prompt_tokens
                        self.display.stats.completion_tokens = completion_tokens
                        self.display.summary(turn, time.monotonic() - start)
                    plan = getattr(self.tool_ctx, "active_plan", None) or {}
                    result.success = True
                    result.final_text = (
                        f"Plan approved: {plan.get('plan') or plan.get('problem', '')}"
                    ).strip()
                    result.turns_used = turn
                    result.tokens_used = total_prompt_tokens + completion_tokens
                    result.files_created = sorted(files_created)
                    result.files_edited = sorted(files_edited)
                    result.commands_run = commands_run
                    result.elapsed_s = time.monotonic() - start
                    result.messages = list(self.messages)
                    return result

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    msg = f"{MAX_CONSECUTIVE_ERRORS} consecutive tool failures — stopping."
                    if self.display:
                        self.display.error(msg)
                    result.error = msg
                    result.turns_used = turn
                    result.tokens_used = total_prompt_tokens + completion_tokens
                    result.files_created = sorted(files_created)
                    result.files_edited = sorted(files_edited)
                    result.commands_run = commands_run
                    result.elapsed_s = time.monotonic() - start
                    result.messages = list(self.messages)
                    return result

                # Recall-first enforcement: warn after MAX_RECALL_SKIP_TURNS consecutive reads
                # without using recall (tracked persistently across turns), refuse on the next read.
                if self.config.permission_mode == "plan":
                    # Update persistent counter with local reads from this turn
                    self.consecutive_reads_without_recall += local_read_without_recall
                    
                    if self.consecutive_reads_without_recall >= MAX_RECALL_SKIP_TURNS:
                        # Warn once per turn (when we first hit the threshold)
                        if self.consecutive_reads_without_recall == MAX_RECALL_SKIP_TURNS:
                            warning_msg = (
                                f"You've called read tools {MAX_RECALL_SKIP_TURNS} times without using `recall`. "
                                "In plan mode, you MUST use `recall(query=...)` first to navigate the codebase. "
                                "The index at `.squishy/index.json` enables efficient file lookup."
                            )
                            if self.display:
                                self.display.warn(warning_msg)
                        # Refuse further reads until recall is used
                        msg = (
                            f"Too many read calls without `recall`. Call `recall(query=...)` now "
                            f"to find relevant files, or call `plan_task` if you have enough information. "
                            "Do not call read_file, list_directory, or search_files again until you use recall."
                        )
                        self.messages.append(
                            {
                                "role": "user",
                                "content": f"[system] {msg}",
                            }
                        )
                        self.consecutive_reads_without_recall = 0  # Reset after enforcing
                        continue

            # Plan-mode enforcement: if the model keeps reading without calling
            # plan_task, nudge it after MAX_PLAN_TOOL_TURNS consecutive tool-call
            # turns.  The nudge budget is shared with the prose-only path above.
            if self.config.permission_mode == "plan":
                active_plan = getattr(self.tool_ctx, "active_plan", None)
                if plan_task_called_this_turn:
                    # plan_task was attempted (approved, declined, or errored).
                    # Either way, reset the counter — the model is trying to plan.
                    turns_without_plan_task = 0
                elif active_plan is None:
                    # No plan exists yet and plan_task wasn't called this turn:
                    # the model spent the turn on read-only investigation.
                    turns_without_plan_task += 1
                    if turns_without_plan_task >= MAX_PLAN_TOOL_TURNS:
                        if plan_nudges < MAX_PLAN_NUDGES:
                            plan_nudges += 1
                            turns_without_plan_task = 0
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[system] You have investigated enough. Stop calling "
                                        "read tools and call `plan_task` now. Use the evidence "
                                        "you already have instead of waiting for exhaustive "
                                        "research. `files_to_modify` and `files_to_create` may "
                                        "be partial or empty if uncertain. Do not read any more "
                                        "files before calling `plan_task`."
                                    ),
                                }
                            )
                            continue
                        else:
                            msg = "plan-mode run finished without producing a plan_task"
                            if self.display:
                                self.display.error(msg)
                            result.error = msg
                            result.turns_used = turn
                            result.tokens_used = total_prompt_tokens + completion_tokens
                            result.elapsed_s = time.monotonic() - start
                            result.messages = list(self.messages)
                            return result

        msg = f"max turns ({self.config.max_turns}) reached"
        if self.display:
            self.display.warn(msg)
            self.display.stats.prompt_tokens = total_prompt_tokens
            self.display.stats.completion_tokens = completion_tokens
            self.display.summary(self.config.max_turns, time.monotonic() - start)
        result.error = msg
        result.turns_used = self.config.max_turns
        result.tokens_used = total_prompt_tokens + completion_tokens
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

        if tc.name == "run_command" and self.display:
            self.display.command_line(str(tc.args.get("command", "")))

        if tc.name == "edit_file" and self.display:
            old_str = str(tc.args.get("old_str", ""))
            new_str = str(tc.args.get("new_str", ""))
            if old_str and new_str:
                self.display.edit_diff(str(tc.args.get("path", "")), old_str, new_str)

        t0 = time.monotonic()
        outcome = await dispatch(tc.name, tc.args, self.tool_ctx, prompt_fn=self.prompt_fn)
        dt_ms = (time.monotonic() - t0) * 1000

        plan_approved = False
        if outcome.success and tc.name == "plan_task":
            # Ask user to approve the plan (even when there's no display).
            if self.display:
                self.display.plan_panel(outcome.data)
            approved = True
            if self.prompt_fn is not None:
                try:
                    from squishy.tools.base import Tool

                    approved = await self.prompt_fn(
                        Tool(name="plan_task", description="", parameters={},
                             run=lambda *_: None),  # type: ignore[arg-type]
                        tc.args,
                    )
                except (EOFError, KeyboardInterrupt):
                    approved = False
            if approved:
                plan = getattr(self.tool_ctx, "active_plan", None)
                if isinstance(plan, dict):
                    plan["approved"] = True
                plan_approved = True
                outcome = ToolResult(
                    True,
                    data={**outcome.data, "approved": True},
                    display=outcome.display,
                )
            else:
                if hasattr(self.tool_ctx, "active_plan"):
                    del self.tool_ctx.active_plan  # type: ignore[attr-defined]
                outcome = ToolResult(
                    False, error="Plan declined by user. Ask for changes or a new approach."
                )

        if self.display:
            if tc.name == "plan_task":
                # Panel already rendered above; no plain tool_result line.
                pass
            elif outcome.success and tc.name == "update_plan":
                plan = getattr(self.tool_ctx, "active_plan", None)
                if plan:
                    self.display.plan_progress(plan.get("steps", []))
            else:
                self.display.tool_result(
                    outcome.success, outcome.display or outcome.error, dt_ms
                )

            if outcome.success and tc.name == "write_file":
                self.display.write_preview(
                    str(tc.args.get("path", "?")), str(tc.args.get("content", ""))
                )
            if outcome.success and tc.name == "run_command":
                self.display.command_output(outcome.data)
            if outcome.success:
                if tc.name == "write_file":
                    self.display.stats.files_created.add(str(tc.args.get("path", "?")))
                elif tc.name == "edit_file":
                    self.display.stats.files_edited.add(str(tc.args.get("path", "?")))
                elif tc.name == "run_command":
                    self.display.stats.commands_run += 1

        self._append_tool_result(tc, message=outcome.to_message())
        return {"success": outcome.success, "plan_approved": plan_approved}
 
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
    # run_command brief is empty; the full command is shown via display.command_line()
    return ""
