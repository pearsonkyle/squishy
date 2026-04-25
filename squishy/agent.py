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
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from squishy.client import Client, CompletionResult, ToolCall
from squishy.config import Config
from squishy.context import build_system_prompt, compact_messages, detect_project, trim_history
from squishy.display import Display, estimate_tokens
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.index.store import has_index
from squishy.plan_state import (
    clear_plan,
    load_plan,
    render_plan_status,
    save_plan,
)
from squishy.quality import assess_response, build_correction
from squishy.tools import PromptFn, ToolContext, ToolResult, REGISTRY, dispatch, openai_schemas
from squishy.tools.scratchpad import render_notes

log = logging.getLogger("squishy.agent")


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
    plan_state: dict[str, Any] | None = None
    empty_responses: int = 0
    quality_skips: int = 0
    prose_completions: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    env_fix_files: list[str] = field(default_factory=list)
    edit_failures: int = 0
    # Phase/budget diagnostics.
    final_phase: str = ""
    explore_turns: int = 0
    fix_verify_cycles: int = 0
    total_quality_violations: int = 0
    # Per-turn event log for post-hoc analysis.
    turn_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _LoopState:
    """Mutable counters shared across the agent loop and its sub-methods."""
    start: float
    consecutive_errors: int = 0
    plan_nudges: int = 0
    turns_without_plan_task: int = 0
    total_prompt_tokens: int = 0
    completion_tokens: int = 0
    files_created: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    commands_run: int = 0
    quality_retries: int = 0
    total_quality_violations: int = 0
    turns_without_progress: int = 0
    test_passed_after_edit: bool = False
    empty_responses: int = 0
    quality_skips: int = 0
    total_tool_calls: dict[str, int] = field(default_factory=dict)
    prose_completions: int = 0
    prior_created_len: int = 0
    prior_edited_len: int = 0
    # Phase-budget tracking (bench/yolo).
    phase: str = "explore"          # "explore" | "fix" | "verify"
    explore_turns: int = 0
    fix_verify_cycles: int = 0
    post_edit_read_turns: int = 0
    finish_countdown: int = -1      # -1 = inactive; N = force finish in N turns
    # Goal-drift tracking (bench/yolo).
    env_fix_files: set[str] = field(default_factory=set)
    problem_files: set[str] = field(default_factory=set)
    env_error_count: int = 0
    # Failed edit tracking.
    edit_failures_per_file: dict[str, int] = field(default_factory=dict)
    total_edit_failures: int = 0
    recent_edit_fail_files: set[str] = field(default_factory=set)  # cleared after successful read
    # Re-anchoring.
    last_reanchor_turn: int = 0
    problem_text: str | None = None
    # Compaction-resilient loop detection: tracks consecutive identical calls
    # independently of message history (which gets trimmed).
    last_call_key: str = ""         # "tool_name:args_hash" of previous dispatch
    consecutive_identical: int = 0  # how many times the same call repeated
    # Per-turn structured event log.
    turn_log: list[dict[str, Any]] = field(default_factory=list)


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
    has_index: bool = field(init=False, default=False)
    session_id: str | None = None
    _last_persisted_idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.tool_ctx = ToolContext(
            working_dir=self.config.working_dir,
            permission_mode=self.config.permission_mode,
            sandbox_image=self.config.sandbox_image,
            use_sandbox=self.config.use_sandbox,
            plan=load_plan(self.config.working_dir),
        )
        self.has_index = has_index(self.config.working_dir)
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

        # Persist initial messages (system prompt) to session.
        self._persist_new_messages()

        self._check_index_staleness()
        if self.display is not None and self.tool_ctx.plan is not None:
            self.display.info(f"[plan] restored {self.tool_ctx.plan.id}")
        if self.display is not None and self.config.permission_mode == "plan" and not self.has_index:
            self.display.info("[plan] no index found; direct file exploration fallback is enabled")

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
        # Reset cross-turn counters for a fresh user turn.
        self.consecutive_reads_without_recall = 0

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

    def _plan_snapshot(self) -> dict[str, Any] | None:
        return self.tool_ctx.plan.to_dict() if self.tool_ctx.plan is not None else None

    def _sync_display_stats(self, st: _LoopState, turn: int) -> None:
        """Update display stats and print summary."""
        if self.display:
            self.display.stats.prompt_tokens = st.total_prompt_tokens
            self.display.stats.completion_tokens = st.completion_tokens
            self.display.summary(turn, time.monotonic() - st.start)

    def _build_result(
        self, st: _LoopState, *, success: bool, final_text: str = "", error: str = "",
        turn: int,
    ) -> TaskResult:
        """Build a TaskResult from the current loop state."""
        # Persist any remaining messages and finalize the session.
        self._persist_new_messages()
        self._finish_session(st, status="completed" if success else "error")
        return TaskResult(
            success=success,
            final_text=final_text,
            error=error,
            turns_used=turn,
            tokens_used=st.total_prompt_tokens + st.completion_tokens,
            files_created=sorted(st.files_created),
            files_edited=sorted(st.files_edited),
            commands_run=st.commands_run,
            elapsed_s=time.monotonic() - st.start,
            messages=list(self.messages),
            plan_state=self._plan_snapshot(),
            empty_responses=st.empty_responses,
            quality_skips=st.quality_skips,
            prose_completions=st.prose_completions,
            tool_call_counts=dict(st.total_tool_calls),
            env_fix_files=sorted(st.env_fix_files),
            edit_failures=st.total_edit_failures,
            final_phase=st.phase,
            explore_turns=st.explore_turns,
            fix_verify_cycles=st.fix_verify_cycles,
            total_quality_violations=st.total_quality_violations,
            turn_log=list(st.turn_log),
        )

    def _persist_new_messages(self) -> None:
        """Persist any new messages since last call to the session store."""
        if not self.session_id:
            return
        new = self.messages[self._last_persisted_idx:]
        if not new:
            return
        try:
            from squishy.session import append_messages
            append_messages(
                self.session_id,
                new,
                root=getattr(self.config, "session_dir", None),
            )
        except Exception:  # noqa: BLE001
            log.debug("session persist failed for %s", self.session_id, exc_info=True)
        self._last_persisted_idx = len(self.messages)

    def _finish_session(self, st: _LoopState, *, status: str = "completed") -> None:
        """Finalize the session with stats."""
        if not self.session_id:
            return
        try:
            from squishy.session import finish_session
            finish_session(
                self.session_id,
                status=status,
                turns=st.turn_log[-1].get("turn", 0) if st.turn_log else 0,
                tokens=st.total_prompt_tokens + st.completion_tokens,
                root=getattr(self.config, "session_dir", None),
            )
        except Exception:  # noqa: BLE001
            log.debug("session finish failed for %s", self.session_id, exc_info=True)

    def _refresh_system_injections(self) -> None:
        """Merge notes and plan-status into the primary system message (index 0).

        Some LLM APIs (e.g. Ollama) only accept a single system message at
        position 0.  Rather than injecting separate system messages, we append
        notes and plan-status blocks to the existing system prompt so there is
        always exactly one system message.

        This method is idempotent: it drops any prior injected blocks before
        re-adding them so content doesn't accumulate across turns.
        """
        if not self.messages:
            return

        # Strip any prior injected blocks (plan-status and notes) from the
        # system message content so we don't accumulate duplicates across turns.
        content = self.messages[0].get("content", "")
        # Remove </notes>...</notes> block
        content = re.sub(
            r"\n<notes>.*?</notes>",
            "",
            content,
            flags=re.DOTALL,
        )
        # Remove plan-status block (starts with "## Plan Status")
        content = re.sub(
            r"\n## Plan Status[\s\S]*?(?=\n\n|\Z)",
            "",
            content,
        )
        # Clean up trailing whitespace/newlines
        content = re.sub(r"\n{3,}", "\n\n", content).rstrip()

        plan = self.tool_ctx.plan
        has_notes = bool(self.tool_ctx.notes)
        has_plan = plan is not None

        if not has_notes and not has_plan:
            # Still update the message in case we stripped content
            self.messages[0]["content"] = content
            return

        # Build the injection blocks
        injection_parts: list[str] = []
        if has_plan:
            injection_parts.append(render_plan_status(plan))
        if has_notes:
            injection_parts.append(render_notes(self.tool_ctx.notes))

        if not injection_parts:
            return

        injection_block = "\n\n".join(injection_parts)
        self.messages[0]["content"] = content + "\n\n" + injection_block

    # ------------------------------------------------------------------
    # Sub-concerns extracted from _run_loop
    # ------------------------------------------------------------------

    def _handle_prose_completion(
        self, completion: CompletionResult, st: _LoopState, turn: int, is_bench: bool,
    ) -> TaskResult | str:
        """Handle a completion with no tool calls.

        Returns a TaskResult to end the run, or a string signal:
          "continue" — a nudge was injected, loop should continue
        """
        in_plan = self.config.permission_mode == "plan"
        plan = self.tool_ctx.plan

        # Plan mode: prose-only not acceptable — agent must call plan_task.
        if in_plan and not is_bench and plan is None:
            if completion.text:
                self.messages.append(_prose_msg(completion.text, completion.reasoning))
            if st.plan_nudges >= self.config.max_plan_nudges:
                msg = "plan-mode run finished without producing a plan_task"
                if self.display:
                    self.display.error(msg)
                return self._build_result(st, success=False, error=msg, turn=turn)
            st.plan_nudges += 1
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
            return "continue"

        # Approved plan with unresolved steps — nudge to continue.
        if (
            not is_bench
            and plan is not None
            and plan.approved
            and self.config.permission_mode != "plan"
            and plan.unresolved_steps()
        ):
            if completion.text:
                self.messages.append(_prose_msg(completion.text, completion.reasoning))
            remaining = "; ".join(
                f"{i + 1}. {step.description}"
                for i, step in enumerate(plan.unresolved_steps()[:4])
            )
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "[system] You have an approved plan with unresolved steps. "
                        "Continue working, then call `update_plan` before finishing. "
                        f"Remaining steps: {remaining}"
                    ),
                }
            )
            return "continue"

        # Empty response — quality failure.
        if not (completion.text or "").strip():
            st.consecutive_errors += 1
            st.empty_responses += 1
            if st.consecutive_errors >= self.config.max_consecutive_errors:
                msg = "model produced empty responses"
                if self.display:
                    self.display.error(msg)
                return self._build_result(st, success=False, error=msg, turn=turn)
            if is_bench and st.consecutive_errors >= 3:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[system] CRITICAL: You have produced multiple empty responses. "
                            "You MUST act NOW. Either:\n"
                            "1. Call `read_file` on the file mentioned in the problem statement, OR\n"
                            "2. Call `edit_file` with your best fix attempt, OR\n"
                            "3. Respond with a plain text summary if you already fixed the bug.\n"
                            "Do NOT produce another empty response."
                        ),
                    }
                )
            else:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[system] Your last response was empty. You must either "
                            "call a tool or respond with a text summary of what you did. "
                            "Continue working on the task."
                        ),
                    }
                )
            return "continue"

        # Normal text-only completion — agent is done.
        st.prose_completions += 1
        if completion.text:
            self.messages.append(_prose_msg(completion.text, completion.reasoning))
        self._sync_display_stats(st, turn)
        return self._build_result(
            st, success=True, final_text=completion.text, turn=turn,
        )

    def _apply_quality_gate(
        self, tool_calls: list[ToolCall], st: _LoopState, turn: int,
    ) -> TaskResult | str | None:
        """Quality gate: catch degenerate tool calls before dispatching.

        Returns:
          TaskResult — forced finish (end the run)
          "skip"    — quality issue detected, correction injected, skip dispatch
          None      — proceed normally
        """
        if self.config.permission_mode not in ("bench", "yolo"):
            return None

        ok, reason = assess_response(tool_calls, self.messages, REGISTRY)
        if ok:
            st.quality_retries = 0
            return None

        st.total_quality_violations += 1

        # Force finish when violations accumulate.
        loop_reasons = ("repeated_tool_call", "excessive_reread", "repeated_command", "edit_verify_loop")
        if reason in loop_reasons:
            # With edits: force-finish at 4 violations.
            if st.files_edited and (st.total_quality_violations >= 4 or st.test_passed_after_edit):
                if self.display:
                    self.display.warn(
                        f"quality: {reason} — force finishing (violations={st.total_quality_violations})"
                    )
                return self._build_result(
                    st, success=True,
                    final_text="Fix applied. The agent completed edits but entered a loop.",
                    turn=turn,
                )
            # Without edits: force-finish at 15 violations to prevent 100-turn loops.
            if not st.files_edited and st.total_quality_violations >= 15:
                if self.display:
                    self.display.warn(
                        f"quality: {reason} — force finishing with no edits (violations={st.total_quality_violations})"
                    )
                return self._build_result(
                    st, success=False,
                    error=f"quality loop: {st.total_quality_violations} violations without edits",
                    turn=turn,
                )

        if st.quality_retries < self.config.max_quality_retries:
            st.quality_retries += 1
            correction = build_correction(reason)
            if self.display:
                self.display.warn(f"quality: {reason}")
            self.messages.append(
                {"role": "user", "content": f"[system] {correction}"}
            )
            st.quality_skips += 1
            return "skip"

        # Exhausted per-turn retries: force finish if edits exist.
        if reason in loop_reasons and st.files_edited:
            if self.display:
                self.display.warn(
                    f"quality: {reason} after edits — finishing"
                )
            return self._build_result(
                st, success=True,
                final_text="Fix applied. The agent completed edits but entered a loop.",
                turn=turn,
            )

        # Other reasons: proceed anyway to avoid deadlock.
        if self.display:
            self.display.warn(f"quality: {reason} (retries exhausted, proceeding)")
        st.quality_retries = 0
        return None

    def _apply_stuck_detection(self, st: _LoopState, is_bench: bool) -> None:
        """Track file-mutation progress and inject stuck nudges in bench mode."""
        made_progress = (
            len(st.files_created) > st.prior_created_len
            or len(st.files_edited) > st.prior_edited_len
        )
        if made_progress:
            st.turns_without_progress = 0
        else:
            st.turns_without_progress += 1
        st.prior_created_len = len(st.files_created)
        st.prior_edited_len = len(st.files_edited)

        if (
            is_bench
            and st.turns_without_progress >= self.config.max_stuck_turns
            and st.turns_without_progress % self.config.max_stuck_turns == 0
        ):
            urgency = (
                "URGENT" if st.turns_without_progress >= self.config.max_stuck_turns * 2
                else "WARNING"
            )
            # List files already read to prevent redundant re-reads.
            already_read = sorted(self.tool_ctx.files_read_count.keys())[:5]
            already_note = (
                f"\nYou have already read: {', '.join(already_read)}. "
                "Do NOT read these files again — use the content you already have."
            ) if already_read else ""

            # On the first stuck nudge, suggest using search if no files were read.
            if not already_read and st.problem_files:
                hint_files = sorted(st.problem_files)[:3]
                hint = f"\nHint: the problem mentions: {', '.join(hint_files)}. Try reading one of those."
            else:
                hint = ""

            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[system] [{urgency}] You have NOT edited any files in "
                        f"{st.turns_without_progress} turns. You MUST call `edit_file` NOW.\n"
                        "Stop reading, searching, and exploring. You have enough information.\n"
                        "1. Pick the most likely file and function from the problem statement.\n"
                        "2. Call `edit_file` with your best fix attempt using content you already have.\n"
                        "A wrong fix that you iterate on is better than more exploration."
                        f"{already_note}{hint}"
                    ),
                }
            )

    def _update_phase(
        self, st: _LoopState, dispatched: list[tuple[ToolCall, dict[str, Any]]],
        *, turn: int = 0,
    ) -> TaskResult | None:
        """Update phase tracking after tool dispatch (bench/yolo only).

        ``dispatched`` is a list of (ToolCall, outcome_dict) for calls that
        were actually dispatched (not blocked by explore_blocked).

        Returns a TaskResult to force-finish, or None to continue.
        """
        if self.config.permission_mode not in ("bench", "yolo"):
            return None

        had_edit = False
        had_test_command = False
        for tc, outcome in dispatched:
            if tc.name == "edit_file" and outcome.get("success"):
                had_edit = True
            elif tc.name == "run_command":
                cmd = str(tc.args.get("command", ""))
                if _is_test_command(cmd):
                    had_test_command = True

        # Phase transitions.
        if had_edit:
            st.phase = "fix"
            st.post_edit_read_turns = 0
        elif had_test_command and st.phase == "fix":
            st.phase = "verify"
            st.fix_verify_cycles += 1
        elif st.phase == "verify":
            st.phase = "fix"

        if st.phase == "explore" and not st.files_edited:
            st.explore_turns += 1

        # Post-edit read-only tracking (bash exploration doesn't count as action).
        if st.files_edited and not had_edit and not had_test_command:
            st.post_edit_read_turns += 1
        elif had_edit or had_test_command:
            st.post_edit_read_turns = 0

        # Budget enforcement.
        if (
            st.phase == "explore"
            and st.explore_turns >= self.config.max_explore_turns
            and not st.files_edited
        ):
            st.phase = "fix"
            self.messages.append({
                "role": "user",
                "content": (
                    f"[system] You have spent {st.explore_turns} turns exploring "
                    "without making any edits. You MUST call `edit_file` NOW with "
                    "your best fix attempt. A wrong fix that you iterate on is "
                    "MUCH better than more exploration."
                ),
            })

        if st.fix_verify_cycles >= self.config.max_fix_verify_cycles and st.files_edited:
            if self.display:
                self.display.warn(
                    f"phase: exhausted fix-verify budget ({st.fix_verify_cycles} cycles)"
                )
            return self._build_result(
                st, success=True,
                final_text="Fix applied. Agent exhausted edit-verify cycle budget.",
                turn=turn,
            )

        # Post-edit read-only warning.
        if st.post_edit_read_turns == self.config.max_post_edit_read_turns:
            self.messages.append({
                "role": "user",
                "content": (
                    f"[system] WARNING: You have spent {st.post_edit_read_turns} "
                    "turns only reading files after making edits. Either:\n"
                    "1. Call `edit_file` with your next fix, OR\n"
                    "2. Call `run_command` to verify your existing fix, OR\n"
                    "3. Respond with text to finish.\n"
                    "Do NOT read more files."
                ),
            })

        return None

    def _inject_turn_budget(self, st: _LoopState, turn: int) -> None:
        """Inject turn-budget awareness every 10 turns (bench/yolo)."""
        if self.config.permission_mode not in ("bench", "yolo"):
            return
        if turn % 10 != 0 or turn == 0:
            return
        remaining = self.config.max_turns - turn
        guidance = {
            "explore": "You should be editing by now. Call edit_file with your best fix.",
            "fix": f"You have made {st.fix_verify_cycles} fix-verify cycles. If tests pass, STOP.",
            "verify": "Verify your fix and finish immediately if it passes.",
        }.get(st.phase, "")
        self.messages.append({
            "role": "user",
            "content": (
                f"[system] Turn {turn}/{self.config.max_turns}. "
                f"{remaining} turns remaining. Phase: {st.phase}. {guidance}"
            ),
        })

    def _inject_test_failure_nudge(
        self, st: _LoopState, tc: ToolCall, outcome: dict[str, Any],
    ) -> None:
        """After repeated test failures, nudge toward a different approach."""
        if self.config.permission_mode not in ("bench", "yolo"):
            return
        if tc.name != "run_command":
            return
        data = outcome.get("data", {})
        if data.get("exit_code", 0) == 0:
            return
        command = str(tc.args.get("command", ""))
        if not any(kw in command for kw in ("pytest", "test", "unittest")):
            return
        if st.fix_verify_cycles < 2:
            return
        if st.fix_verify_cycles >= 4:
            self.messages.append({
                "role": "user",
                "content": (
                    f"[system] CRITICAL: {st.fix_verify_cycles} fix-verify cycles "
                    "and the test still fails. STOP making small tweaks — your "
                    "approach is fundamentally wrong. You MUST:\n"
                    "1. Re-read the failing test to understand exactly what it expects.\n"
                    "2. Re-read the problem statement to check your understanding.\n"
                    "3. Try a COMPLETELY different fix strategy.\n"
                    "If you cannot fix it, respond with a text summary and stop."
                ),
            })
        else:
            self.messages.append({
                "role": "user",
                "content": (
                    f"[system] This test has failed after {st.fix_verify_cycles} fix "
                    "attempts. Before editing again:\n"
                    "1. Re-read the test file to understand what it actually expects.\n"
                    "2. Check if you are editing the wrong function or class.\n"
                    "3. Consider reverting and trying a completely different fix."
                ),
            })

    def _check_goal_drift(
        self, st: _LoopState, tc: ToolCall, outcome: dict[str, Any],
    ) -> None:
        """Detect when the agent is fixing environmental issues instead of the bug."""
        if self.config.permission_mode not in ("bench", "yolo"):
            return

        if tc.name == "edit_file" and outcome.get("success"):
            path = str(tc.args.get("path", ""))
            if st.problem_files and not _path_matches_problem(path, st.problem_files):
                st.env_fix_files.add(path)

        if tc.name == "run_command" and not outcome.get("success"):
            data = outcome.get("data", {})
            output = str(data.get("stderr", "")) + str(data.get("stdout", ""))
            env_patterns = (
                "ImportError", "ModuleNotFoundError", "No module named",
                "cannot import name", "collections.Mapping",
                "collections.MutableMapping", "collections.Callable",
            )
            if any(p in output for p in env_patterns):
                st.env_error_count += 1

        should_nudge = (
            (len(st.env_fix_files) >= 2 and st.env_error_count >= 2)
            or st.env_error_count >= 3
        )
        if should_nudge:
            st.env_error_count = 0  # reset to avoid spam
            self.messages.append({
                "role": "user",
                "content": (
                    "[system] GOAL DRIFT WARNING: You are fixing environmental/import "
                    "errors instead of the actual bug described in the problem statement. "
                    "These import errors are caused by Python version differences in the "
                    "test environment — they are NOT the bug you need to fix.\n\n"
                    "STOP fixing import compatibility issues. Instead:\n"
                    "1. Focus ONLY on the bug described in the problem statement.\n"
                    "2. Run a more targeted test: "
                    "`python -m pytest path/to/test.py::specific_test -x`\n"
                    "3. Or write a minimal reproduction script to verify your fix.\n\n"
                    "Re-read the problem statement and get back on track."
                ),
            })

    def _track_edit_failure(
        self, st: _LoopState, tc: ToolCall, outcome: dict[str, Any],
    ) -> None:
        """Track edit_file failures per file and nudge on repeated failures."""
        if self.config.permission_mode not in ("bench", "yolo"):
            return
        if tc.name != "edit_file":
            return

        path = str(tc.args.get("path", ""))
        if not path:
            return

        if not outcome.get("success"):
            st.edit_failures_per_file[path] = st.edit_failures_per_file.get(path, 0) + 1
            st.total_edit_failures += 1
            failures = st.edit_failures_per_file[path]

            if failures == 3:
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[system] You have failed to edit `{path}` {failures} times. "
                        "Your old_str is not matching the file content. STOP guessing and:\n"
                        "1. Call `read_file` on the exact line range you want to edit.\n"
                        "2. Copy the EXACT text from the read output into old_str.\n"
                        "3. Include 2-3 lines of surrounding context for uniqueness."
                    ),
                })
            elif failures >= 5:
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[system] CRITICAL: {failures} failed edits to `{path}`. "
                        "You are struggling with this file. Consider:\n"
                        "1. Are you editing the RIGHT file? Re-read the problem statement.\n"
                        "2. Try a completely different approach or a different file.\n"
                        "3. Write a small reproduction script to verify you understand "
                        "the bug before editing."
                    ),
                })
        else:
            st.edit_failures_per_file[path] = 0

    def _maybe_reanchor_problem(self, st: _LoopState, turn: int) -> None:
        """Periodically re-inject the problem statement to prevent drift."""
        if self.config.permission_mode not in ("bench", "yolo"):
            return
        reanchor_interval = 15
        if turn < reanchor_interval or turn - st.last_reanchor_turn < reanchor_interval:
            return

        # Use cached problem text if available; otherwise scan and cache.
        if st.problem_text is None:
            for msg in self.messages:
                if msg.get("role") == "user" and not str(msg.get("content", "")).startswith("[system]"):
                    content = str(msg.get("content", ""))
                    if "## Problem" in content:
                        text = content.split("## Problem", 1)[1].split("## Hints")[0].strip()
                        if len(text) > 500:
                            text = text[:500] + "..."
                        st.problem_text = text
                    break

        if not st.problem_text:
            return

        st.last_reanchor_turn = turn
        self.messages.append({
            "role": "user",
            "content": (
                f"[system] REMINDER — The original bug you are fixing:\n"
                f"{st.problem_text}\n\n"
                "Stay focused on THIS bug. Do not fix unrelated issues."
            ),
        })

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self, start: float) -> TaskResult:
        st = _LoopState(start=start)
        is_bench = self.config.permission_mode == "bench"
        _is_constrained = self.config.permission_mode in ("bench", "yolo")

        # Extract problem file paths for goal-drift detection (bench/yolo).
        if _is_constrained:
            for msg in self.messages:
                if msg.get("role") == "user" and not str(msg.get("content", "")).startswith("[system]"):
                    st.problem_files = _extract_problem_files(str(msg.get("content", "")))
                    break

        _cached_perm_mode = self.config.permission_mode
        _cached_schemas = openai_schemas(_cached_perm_mode)

        for turn in range(1, self.config.max_turns + 1):
            # Finish countdown: force-finish if the model keeps calling tools
            # after a test passed.
            if _is_constrained and st.finish_countdown >= 0:
                if st.finish_countdown == 0:
                    if self.display:
                        self.display.warn("finish countdown expired — force finishing")
                    return self._build_result(
                        st, success=True,
                        final_text="Fix applied and verified. Agent did not stop after test passed.",
                        turn=turn,
                    )
                st.finish_countdown -= 1

            self.tool_ctx.permission_mode = self.config.permission_mode
            if self.config.permission_mode != _cached_perm_mode:
                _cached_perm_mode = self.config.permission_mode
                _cached_schemas = openai_schemas(_cached_perm_mode)
            schemas = _cached_schemas

            # Re-inject fresh system messages each turn.
            if not is_bench:
                self._refresh_system_injections()

            # Layer 2 compaction + trim (trim_history calls snip_old_tool_results).
            msg_count_before = len(self.messages)
            if getattr(self.client, "context_window", 0) > 0:
                self.messages[:] = await compact_messages(
                    self.messages,
                    self.client,
                    context_limit=self.client.context_window,
                    threshold=self.config.compaction_threshold,
                )
            self.messages[:] = trim_history(
                self.messages, max_messages=self.config.max_history_messages
            )
            # Reset persistence index after compaction/trim shrinks the list.
            self._last_persisted_idx = len(self.messages)

            # After compaction/trim, inject a note about files already read
            # so the model doesn't blindly re-read them.
            if len(self.messages) < msg_count_before and self.tool_ctx.files_read_count:
                read_files = sorted(self.tool_ctx.files_read_count.keys())
                if len(read_files) > 15:
                    read_files = read_files[:15] + [f"... and {len(read_files) - 15} more"]
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[system] Context was compacted. Files you already read in this session: "
                        f"{', '.join(read_files)}. "
                        "If you need to edit a file but don't remember the exact content for "
                        "old_str, call read_file again to get the precise text — do NOT guess. "
                        "Use save_note to persist important content across compactions."
                    ),
                })

            # LLM call.
            try:
                completion = await self.client.complete(
                    self.messages, schemas, stream=True, on_text=self._on_text,
                )
            except LLMError as e:
                if self.display:
                    self.display.error(f"LLM error: {e}")
                return self._build_result(
                    st, success=False, error=str(e), turn=turn - 1,
                )

            if completion.text and self.display:
                self.display.console.print()

            st.total_prompt_tokens += completion.prompt_tokens
            st.completion_tokens += completion.completion_tokens

            # --- No tool calls: prose-only completion ---
            if not completion.tool_calls:
                result = self._handle_prose_completion(completion, st, turn, is_bench)
                if isinstance(result, TaskResult):
                    return result
                # result == "continue"
                continue

            self.messages.append(_assistant_msg(completion.text, completion.tool_calls, completion.reasoning))

            # --- Compaction-resilient loop detection (bench/yolo only) ---
            # Build a key from all tool calls this turn and compare to previous.
            # Skip when explore_blocked is active to avoid force-terminating
            # on intentionally blocked retries (explore_blocked computed below).
            explore_blocked = False
            if _is_constrained:
                call_key = _call_key(completion.tool_calls)
                if call_key == st.last_call_key:
                    st.consecutive_identical += 1
                else:
                    st.consecutive_identical = 0
                    st.last_call_key = call_key

            if _is_constrained and st.consecutive_identical >= 7:
                # Force-finish: model is stuck repeating the same call.
                # Threshold raised from 4 to 7 to give nudge messages (at 2)
                # more time to redirect the model.
                msg = (f"loop detected: same tool call repeated "
                       f"{st.consecutive_identical + 1} times consecutively")
                if self.display:
                    self.display.warn(msg)
                return self._build_result(
                    st, success=bool(st.files_edited),
                    error="" if st.files_edited else msg,
                    final_text="Fix applied." if st.files_edited else "",
                    turn=turn,
                )
            if _is_constrained and st.consecutive_identical >= 2:
                if st.consecutive_identical >= 5:
                    # Escalated warning — model ignored initial nudge.
                    hint_files = sorted(st.problem_files)[:3] if st.problem_files else []
                    hint = (
                        f"\nThe problem mentions these files: {', '.join(hint_files)}. "
                        "If you haven't edited one yet, do so NOW."
                    ) if hint_files else ""
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"[system] CRITICAL: You have repeated the EXACT same call "
                            f"{st.consecutive_identical + 1} times. ONE MORE and this task "
                            "will be terminated. You MUST do something DIFFERENT right now:\n"
                            "- Call `edit_file` with your best guess fix.\n"
                            "- Or respond with text to finish."
                            f"{hint}"
                        ),
                    })
                else:
                    # Initial warning after 3 consecutive identical calls.
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"[system] WARNING: You have made the EXACT same tool call "
                            f"{st.consecutive_identical + 1} times in a row. You are stuck "
                            "in a loop. You MUST try something different:\n"
                            "1. If you need to edit a file, call `edit_file` now.\n"
                            "2. If you already fixed the bug, respond with plain text.\n"
                            "3. Try a completely different file or approach.\n"
                            "Do NOT repeat the same call again."
                        ),
                    })

            # --- Dispatch tools ---
            plan_task_called_this_turn = False
            local_read_without_recall = 0
            dispatched_pairs: list[tuple[ToolCall, dict[str, Any]]] = []

            # Bench/yolo: block read-only tools after prolonged no-edit turns,
            # or after too many read-only turns post-edit.
            # Allow one read-only call through every 3 blocked turns so the
            # model can still make incremental progress toward finding the
            # right file instead of being permanently stuck.
            _explore_eligible = _is_constrained and (
                (not st.files_edited
                 and st.turns_without_progress >= self.config.max_stuck_turns * 2)
                or (st.files_edited
                    and st.post_edit_read_turns >= self.config.max_post_edit_read_turns + 2)
            )
            # Let one call through every 3 turns to avoid permanent stall.
            _blocked_streak = (
                st.turns_without_progress - self.config.max_stuck_turns * 2
                if not st.files_edited
                else st.post_edit_read_turns - self.config.max_post_edit_read_turns - 2
            )
            explore_blocked = _explore_eligible and (_blocked_streak % 3 != 0)

            # --- Quality gate (skip when explore blocker is active to avoid
            #     spurious repeated_tool_call violations from blocked retries) ---
            if not explore_blocked:
                gate = self._apply_quality_gate(completion.tool_calls, st, turn)
                if isinstance(gate, TaskResult):
                    return gate
                if gate == "skip":
                    continue

            for tc in completion.tool_calls:
                if tc.name == "plan_task":
                    plan_task_called_this_turn = True
                st.total_tool_calls[tc.name] = st.total_tool_calls.get(tc.name, 0) + 1

                # If exploration is blocked, refuse read-only tools AND
                # bash exploration commands (grep/sed/cat) that bypass the blocker.
                # Exception: allow read_file on files with recent edit failures.
                if explore_blocked:
                    blocked = False
                    if tc.name in _EXPLORE_TOOLS:
                        # Allow reading a file that just had an edit failure.
                        if tc.name == "read_file" and str(tc.args.get("path", "")) in st.recent_edit_fail_files:
                            blocked = False
                        else:
                            blocked = True
                    elif tc.name == "run_command" and _is_exploration_command(
                        str(tc.args.get("command", ""))
                    ):
                        blocked = True
                    if blocked:
                        self._append_tool_result(
                            tc,
                            message=(
                                '{"success": false, "error": "Exploration blocked: you have spent '
                                f'{st.turns_without_progress} turns reading/searching without making '
                                'any edits. You MUST call edit_file NOW with your best fix attempt. '
                                'Use the content you already have. A wrong fix that you iterate on '
                                'is better than more exploration."}'
                            ),
                        )
                        continue

                outcome = await self._run_tool(turn, tc)
                dispatched_pairs.append((tc, outcome))

                # Test failure nudge (bench/yolo).
                self._inject_test_failure_nudge(st, tc, outcome)
                self._check_goal_drift(st, tc, outcome)
                self._track_edit_failure(st, tc, outcome)

                if outcome["success"]:
                    st.consecutive_errors = 0
                    if tc.name in ("read_file", "list_directory", "search_files"):
                        local_read_without_recall += 1
                        # Clear edit-fail exemption after successful re-read.
                        if tc.name == "read_file":
                            st.recent_edit_fail_files.discard(str(tc.args.get("path", "")))
                    elif tc.name == "recall":
                        local_read_without_recall = 0
                        self.consecutive_reads_without_recall = 0
                    if tc.name == "write_file":
                        st.files_created.add(str(tc.args.get("path", "?")))
                    elif tc.name == "edit_file":
                        # Detect no-op edits (old_str == new_str).
                        old = str(tc.args.get("old_str", ""))
                        new = str(tc.args.get("new_str", ""))
                        if old and old == new:
                            st.total_quality_violations += 1
                        else:
                            st.files_edited.add(str(tc.args.get("path", "?")))
                    elif tc.name == "run_command":
                        st.commands_run += 1
                        # Track test-passed-after-edit for force-finish logic.
                        # Only trigger on actual test commands, not pwd/ls/grep.
                        cmd = str(tc.args.get("command", ""))
                        if (
                            st.files_edited
                            and outcome.get("data", {}).get("exit_code") == 0
                            and _is_test_command(cmd)
                        ):
                            st.test_passed_after_edit = True
                elif tc.name == "run_command":
                    st.commands_run += 1
                    st.consecutive_errors = 0
                else:
                    st.consecutive_errors += 1
                    # After edit_file failure, allow re-reading the target file
                    # so the agent can get exact content for old_str.
                    if tc.name == "edit_file":
                        edit_path = str(tc.args.get("path", ""))
                        if edit_path:
                            st.recent_edit_fail_files.add(edit_path)
                            if edit_path in self.tool_ctx.files_read_count:
                                self.tool_ctx.files_read_count[edit_path] = min(
                                    self.tool_ctx.files_read_count[edit_path], 2
                                )

                # Plan-approved terminal event.
                if (
                    outcome.get("plan_approved")
                    and self.config.permission_mode == "plan"
                ):
                    self._sync_display_stats(st, turn)
                    plan = self.tool_ctx.plan.to_dict() if self.tool_ctx.plan is not None else {}
                    return self._build_result(
                        st, success=True,
                        final_text=f"Plan approved: {plan.get('problem', '')}".strip(),
                        turn=turn,
                    )

                if st.consecutive_errors >= self.config.max_consecutive_errors:
                    msg = f"{self.config.max_consecutive_errors} consecutive tool failures — stopping."
                    if self.display:
                        self.display.error(msg)
                    return self._build_result(st, success=False, error=msg, turn=turn)

                # Recall-first enforcement (plan mode only).
                if self.config.permission_mode == "plan" and not is_bench:
                    recall_skip_budget = self.config.max_recall_skip_turns
                    self.consecutive_reads_without_recall += local_read_without_recall

                    if self.has_index and self.consecutive_reads_without_recall >= recall_skip_budget:
                        if self.consecutive_reads_without_recall == recall_skip_budget:
                            warning_msg = (
                                f"You've called read tools {recall_skip_budget} times without using `recall`. "
                                "In plan mode, you MUST use `recall(query=...)` first to navigate the codebase. "
                                "The index at `.squishy/index.json` enables efficient file lookup."
                            )
                            if self.display:
                                self.display.warn(warning_msg)
                        self.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[system] Too many read calls without `recall`. Call `recall(query=...)` now "
                                    "to find relevant files, or call `plan_task` if you have enough information. "
                                    "Do not call read_file, list_directory, or search_files again until you use recall."
                                ),
                            }
                        )
                        self.consecutive_reads_without_recall = 0
                        # Break the inner tool loop and restart the outer turn loop.
                        break

            # Plan-mode investigation nudge.
            if self.config.permission_mode == "plan" and not is_bench:
                active_plan = self.tool_ctx.plan
                if plan_task_called_this_turn:
                    st.turns_without_plan_task = 0
                elif active_plan is None:
                    st.turns_without_plan_task += 1
                    if st.turns_without_plan_task >= self.config.max_plan_investigation_turns:
                        if st.plan_nudges < self.config.max_plan_nudges:
                            st.plan_nudges += 1
                            st.turns_without_plan_task = 0
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
                            return self._build_result(st, success=False, error=msg, turn=turn)

            # Phase tracking and budgets (bench/yolo).
            phase_result = self._update_phase(st, dispatched_pairs, turn=turn)
            if isinstance(phase_result, TaskResult):
                return phase_result

            # Bench/yolo: if a test passed after edits, nudge agent to finish
            # and start a 2-turn countdown.
            if _is_constrained and st.test_passed_after_edit:
                st.test_passed_after_edit = False  # consume the flag
                st.finish_countdown = 1
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[system] A test/verification command passed after your edits. "
                            "Your fix is working. You are DONE. Respond with ONLY a plain text "
                            "summary of what you changed and why. Do NOT call any more tools. "
                            "Do NOT run more commands. Just write text and stop."
                        ),
                    }
                )

            # Soft cap: no edits after 50 turns → force finish (bench only).
            # Previously 30 turns, but analysis showed models that are slow
            # to identify the right file DO eventually succeed if given time.
            if is_bench and turn >= 50 and not st.files_edited:
                msg = f"no edits after {turn} turns — force finishing"
                if self.display:
                    self.display.warn(msg)
                return self._build_result(
                    st, success=False, error=msg, turn=turn,
                )

            # Turn budget injection (bench/yolo).
            self._inject_turn_budget(st, turn)
            self._maybe_reanchor_problem(st, turn)

            # Stuck detection.
            self._apply_stuck_detection(st, is_bench)

            # Per-turn event log (bench/yolo only — keeps transcripts diagnostic).
            if _is_constrained:
                tools_this_turn = [
                    {"name": tc.name, "args_summary": _brief(tc) or str(tc.args.get("command", ""))[:80]}
                    for tc in completion.tool_calls
                ]
                st.turn_log.append({
                    "turn": turn,
                    "phase": st.phase,
                    "tools": tools_this_turn,
                    "dispatched": len(dispatched_pairs),
                    "blocked": len(completion.tool_calls) - len(dispatched_pairs),
                    "consecutive_identical": st.consecutive_identical,
                    "quality_violations": st.total_quality_violations,
                    "files_edited": len(st.files_edited),
                    "fix_verify_cycles": st.fix_verify_cycles,
                    "explore_turns": st.explore_turns,
                    "elapsed_s": round(time.monotonic() - st.start, 1),
                })

            # Persist messages accumulated this turn to the session store.
            self._persist_new_messages()

        # Max turns exhausted.
        msg = f"max turns ({self.config.max_turns}) reached"
        if self.display:
            self.display.warn(msg)
            self._sync_display_stats(st, self.config.max_turns)
        return self._build_result(st, success=False, error=msg, turn=self.config.max_turns)

    async def _handle_plan_approval(
        self, tc: ToolCall, outcome: ToolResult,
    ) -> tuple[ToolResult, bool]:
        """Handle plan_task approval flow. Returns (outcome, plan_approved)."""
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
            if self.tool_ctx.plan is not None:
                self.tool_ctx.plan.mark_approved()
                self.tool_ctx.plan_switch_prompted = False
                save_plan(self.tool_ctx.working_dir, self.tool_ctx.plan)
            outcome = ToolResult(
                True,
                data={
                    **outcome.data,
                    "approved": True,
                    "plan": self.tool_ctx.plan.to_dict() if self.tool_ctx.plan is not None else {},
                },
                display=outcome.display,
            )
            return outcome, True
        self.tool_ctx.plan = None
        self.tool_ctx.pending_plan_evidence.clear()
        self.tool_ctx.plan_switch_prompted = False
        clear_plan(self.tool_ctx.working_dir)
        return ToolResult(False, error="Plan declined by user. Ask for changes or a new approach."), False

    def _record_plan_evidence(self, tc: ToolCall, outcome: ToolResult) -> None:
        """Record tool outcome as plan evidence when an approved plan is active."""
        exit_code = outcome.data.get("exit_code")
        ran_command = tc.name == "run_command" and exit_code is not None
        if not (outcome.success or ran_command):
            return
        if self.tool_ctx.plan is None or not self.tool_ctx.plan.approved:
            return
        if tc.name in ("write_file", "edit_file"):
            self.tool_ctx.pending_plan_evidence.append({
                "kind": tc.name,
                "path": str(tc.args.get("path", "")),
                "detail": "created or rewrote file" if tc.name == "write_file" else "edited existing file",
            })
        elif tc.name == "run_command":
            data = outcome.data
            self.tool_ctx.pending_plan_evidence.append({
                "kind": "run_command",
                "command": str(tc.args.get("command", "")),
                "exit_code": int(exit_code) if isinstance(exit_code, int) else None,
                "detail": str(data.get("stderr") or data.get("stdout") or "").strip()[:300],
            })

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
            outcome, plan_approved = await self._handle_plan_approval(tc, outcome)

        self._record_plan_evidence(tc, outcome)

        if self.display:
            if tc.name == "plan_task":
                pass  # Panel already rendered in _handle_plan_approval.
            elif outcome.success and tc.name == "update_plan":
                plan = self.tool_ctx.plan
                if plan:
                    self.display.plan_progress([step.to_dict() for step in plan.steps])
            else:
                self.display.tool_result(
                    outcome.success, outcome.display or outcome.error, dt_ms
                )

            if outcome.success and tc.name == "write_file":
                self.display.write_preview(
                    str(tc.args.get("path", "?")), str(tc.args.get("content", ""))
                )
            if tc.name == "run_command" and outcome.data.get("exit_code") is not None:
                self.display.command_output(outcome.data)
            if outcome.success:
                if tc.name == "write_file":
                    self.display.stats.files_created.add(str(tc.args.get("path", "?")))
                elif tc.name == "edit_file":
                    self.display.stats.files_edited.add(str(tc.args.get("path", "?")))
                elif tc.name == "run_command":
                    self.display.stats.commands_run += 1

        self._append_tool_result(tc, message=outcome.to_message())

        # Semantic anchoring: tag important tool results so they survive
        # history trimming.
        if self.messages and self.messages[-1].get("role") == "tool":
            should_anchor = (
                (tc.name == "run_command" and not outcome.success)
                or (tc.name == "edit_file" and outcome.success)
                or (tc.name == "search_files" and outcome.success and outcome.data.get("count", 0) > 0)
                or (tc.name == "read_file" and outcome.success
                    and self.tool_ctx.files_read_count.get(str(tc.args.get("path", "")), 0) <= 1)
            )
            if should_anchor:
                self.messages[-1]["_squishy_anchor"] = True

        return {
            "success": outcome.success,
            "plan_approved": plan_approved,
            "data": outcome.data if isinstance(outcome.data, dict) else {},
        }

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


def _prose_msg(text: str, reasoning: str = "") -> dict[str, Any]:
    """Build a prose-only assistant message, preserving reasoning if present."""
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        msg["think"] = reasoning
    return msg


def _assistant_msg(
    text: str, tool_calls: list[ToolCall], reasoning: str = "",
) -> dict[str, Any]:
    msg: dict[str, Any] = {
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
    # Preserve reasoning/thinking for session persistence and training data.
    # This key is ignored by the OpenAI API but survives in self.messages.
    if reasoning:
        msg["think"] = reasoning
    return msg


def _brief(tc: ToolCall) -> str:
    a = tc.args
    if tc.name in ("read_file", "write_file", "edit_file", "list_directory"):
        return str(a.get("path", ""))
    if tc.name == "search_files":
        return f'"{a.get("pattern", "")}"'
    if tc.name == "glob_files":
        return str(a.get("pattern", ""))
    if tc.name == "recall":
        return str(a.get("query", ""))
    # run_command brief is empty; the full command is shown via display.command_line()
    return ""


_EXPLORE_TOOLS = frozenset({"read_file", "list_directory", "search_files", "glob_files"})
_TEST_CMD_KEYWORDS = ("pytest", "unittest", "python -m test", "python -m pytest", "test_")


def _is_test_command(cmd: str) -> bool:
    """Return True if ``cmd`` looks like a test invocation (not ls/pwd/grep)."""
    return any(kw in cmd for kw in _TEST_CMD_KEYWORDS)


def _is_exploration_command(cmd: str) -> bool:
    """Return True if ``cmd`` is a read-only exploration command (grep/sed/cat/find)."""
    first = cmd.strip().split()[0] if cmd.strip() else ""
    return first in ("grep", "rg", "sed", "cat", "head", "tail", "find", "awk", "wc", "od")


# Regex for Python file paths like  foo/bar/baz.py  or  foo/bar.py
_PY_PATH_RE = re.compile(r"(?:^|[\s\"'`(,])([a-zA-Z_][\w/]*\.py)\b")
# Regex for dotted module paths like  sympy.core.power  or  django.core.checks
_MODULE_RE = re.compile(r"(?:^|[\s\"'`(,])([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*){2,})\b")


def _extract_problem_files(text: str) -> set[str]:
    """Extract likely file paths and module references from a problem statement.

    Returns a set of lowercased partial paths (e.g., ``{'sympy/core/power.py',
    'astropy/modeling/separable.py'}``).  Used for goal-drift heuristics — does
    not need to be perfectly accurate.
    """
    paths: set[str] = set()
    for m in _PY_PATH_RE.finditer(text):
        paths.add(m.group(1).lower())
    for m in _MODULE_RE.finditer(text):
        parts = m.group(1).split(".")
        # module.submodule.name -> module/submodule/name.py + module/submodule.py
        paths.add("/".join(parts).lower() + ".py")
        if len(parts) > 2:
            paths.add("/".join(parts[:-1]).lower() + ".py")
    return paths


def _call_key(tool_calls: list[ToolCall]) -> str:
    """Build a stable key from a list of tool calls for loop detection."""
    parts = []
    for tc in tool_calls:
        try:
            args_str = json.dumps(tc.args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(tc.args)
        parts.append(f"{tc.name}:{args_str}")
    return "|".join(parts)


def _path_matches_problem(path: str, problem_files: set[str]) -> bool:
    """Check if an edited file path plausibly relates to the problem statement."""
    path_lower = path.lower().replace("\\", "/")
    for pf in problem_files:
        if pf in path_lower or path_lower.endswith(pf):
            return True
    # Also check base name overlap.
    base = os.path.basename(path_lower).replace(".py", "")
    return any(base in pf for pf in problem_files)
