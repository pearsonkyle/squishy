"""Async agent loop — slim orchestrator.

Delegates to:
  agent_state.py    — TaskResult, LoopState, message helpers
  agent_safety.py   — loop detection, quality gates, nudges
  agent_dispatch.py — tool dispatch, plan approval, evidence
  agent_phases.py   — phase tracking, re-anchoring, budgets
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

from squishy.agent_dispatch import append_tool_result, run_tool, track_tool_outcome
from squishy.agent_phases import (
    cache_problem_text,
    inject_turn_budget,
    maybe_reanchor_problem,
    update_phase,
)
from squishy.agent_safety import (
    apply_quality_gate,
    apply_stuck_detection,
    check_goal_drift,
    check_read_only_spiral,
    inject_consecutive_identical_nudge,
    inject_nudge,
    inject_test_failure_nudge,
    track_edit_failure,
)
from squishy.agent_state import (
    EXPLORE_TOOLS,
    LoopState,
    TaskResult,
    assistant_msg,
    brief,
    call_key,
    extract_problem_files,
    is_exploration_command,
    prose_msg,
)
from squishy.client import Client, CompletionResult, ToolCall
from squishy.config import Config
from squishy.context import build_system_prompt, compact_messages, detect_project, trim_history
from squishy.display import Display, estimate_tokens
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.index.store import has_index
from squishy.plan_state import load_plan, render_plan_status
from squishy.tools import PromptFn, ToolContext, openai_schemas
from squishy.tools.scratchpad import render_notes

log = logging.getLogger("squishy.agent")


@dataclass
class Agent:
    config: Config
    client: Client
    display: Display | None = None
    prompt_fn: PromptFn | None = None
    tool_ctx: ToolContext = field(init=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    consecutive_reads_without_recall: int = 0
    has_index: bool = field(init=False, default=False)
    session_id: str | None = None
    _last_persisted_idx: int = field(init=False, default=0)
    _full_log: list[dict[str, Any]] = field(init=False, default_factory=list)
    _full_log_idx: int = field(init=False, default=0)

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
        self.messages.append({"role": "system", "content": system_prompt})

        if self.display is not None:
            self.display.stats.prompt_tokens += estimate_tokens(system_prompt)

        self._persist_new_messages()
        self._check_index_staleness()
        if self.display is not None:
            self.display.set_mode(self.config.permission_mode)
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
        self, user_message: str, *, timeout: float | None = None,
    ) -> TaskResult:
        """Run one user turn to completion."""
        self.consecutive_reads_without_recall = 0
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
        except KeyboardInterrupt:
            # Ctrl+C anywhere inside the run — including inside an approval
            # prompt — should abort the whole turn, not just decline the
            # current tool. Translate into our usual cancelled signal so
            # the CLI's outer handler resets cleanly.
            if self.display:
                self.display.flush_streaming_text()
            raise AgentCancelled("interrupted by user") from None

    # ------------------------------------------------------------------
    # Result building and session persistence
    # ------------------------------------------------------------------

    def _plan_snapshot(self) -> dict[str, Any] | None:
        return self.tool_ctx.plan.to_dict() if self.tool_ctx.plan is not None else None

    def _sync_display_stats(self, st: LoopState, turn: int) -> None:
        if self.display:
            self.display.stats.prompt_tokens = st.total_prompt_tokens
            self.display.stats.completion_tokens = st.completion_tokens
            self.display.summary(turn, time.monotonic() - st.start)

    def _build_result(
        self, st: LoopState, *, success: bool, final_text: str = "", error: str = "",
        turn: int,
    ) -> TaskResult:
        # Capture remaining messages before building result.
        remaining = self.messages[self._full_log_idx:]
        if remaining:
            self._full_log.extend(remaining)
            self._full_log_idx = len(self.messages)

        self._persist_new_messages()
        self._finish_session(st, status="completed" if success else "error")
        return TaskResult(
            success=success, final_text=final_text, error=error,
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
            full_log=list(self._full_log),
        )

    def _persist_new_messages(self) -> None:
        if not self.session_id:
            return
        new = self.messages[self._last_persisted_idx:]
        if not new:
            return
        try:
            from squishy.session import append_messages
            append_messages(
                self.session_id, new,
                root=getattr(self.config, "session_dir", None),
            )
        except Exception:  # noqa: BLE001
            log.debug("session persist failed for %s", self.session_id, exc_info=True)
        self._last_persisted_idx = len(self.messages)

    def _finish_session(self, st: LoopState, *, status: str = "completed") -> None:
        if not self.session_id:
            return
        try:
            from squishy.session import finish_session
            finish_session(
                self.session_id, status=status,
                turns=st.turn_log[-1].get("turn", 0) if st.turn_log else 0,
                tokens=st.total_prompt_tokens + st.completion_tokens,
                root=getattr(self.config, "session_dir", None),
            )
        except Exception:  # noqa: BLE001
            log.debug("session finish failed for %s", self.session_id, exc_info=True)

    # ------------------------------------------------------------------
    # System message injection
    # ------------------------------------------------------------------

    def _refresh_system_injections(self) -> None:
        """Merge notes and plan-status into the primary system message (index 0)."""
        if not self.messages:
            return
        content = self.messages[0].get("content", "")
        # Remove <notes>...</notes> block
        content = re.sub(r"\n<notes>.*?</notes>", "", content, flags=re.DOTALL)
        # Remove plan-status block
        content = re.sub(r"\n<plan-status>.*?</plan-status>", "", content, flags=re.DOTALL)
        content = re.sub(r"\n{3,}", "\n\n", content).rstrip()

        plan = self.tool_ctx.plan
        has_notes = bool(self.tool_ctx.notes)
        has_plan = plan is not None

        if not has_notes and not has_plan:
            self.messages[0]["content"] = content
            return

        injection_parts: list[str] = []
        if has_plan:
            injection_parts.append(render_plan_status(plan))
        if has_notes:
            injection_parts.append(render_notes(self.tool_ctx.notes))
        self.messages[0]["content"] = content + "\n\n" + "\n\n".join(injection_parts)

    # ------------------------------------------------------------------
    # Prose completion handling
    # ------------------------------------------------------------------

    def _handle_prose_completion(
        self, completion: CompletionResult, st: LoopState, turn: int, is_bench: bool,
    ) -> TaskResult | str:
        """Handle a completion with no tool calls.

        Returns a TaskResult to end the run, or "continue".
        """
        in_plan = self.config.permission_mode == "plan"
        plan = self.tool_ctx.plan

        # Plan mode: prose-only not acceptable — agent must call plan_task.
        if in_plan and not is_bench and plan is None:
            if completion.text:
                self.messages.append(prose_msg(completion.text, completion.reasoning))
                if self.display:
                    self.display.flush_streaming_text()
            if st.plan_nudges >= self.config.max_plan_nudges:
                msg = "plan-mode run finished without producing a plan_task"
                if self.display:
                    self.display.error(msg)
                return self._build_result(st, success=False, error=msg, turn=turn)
            st.plan_nudges += 1
            # If the model wrote a plan as a JSON literal in prose, call
            # that out specifically — the generic "call plan_task" nudge
            # is easy for weak models to misread as "describe a plan".
            looks_like_json_plan = _looks_like_json_plan(completion.text or "")
            if looks_like_json_plan:
                content = (
                    "[system] You wrote a JSON plan inside your message, but "
                    "that does NOT count as planning. The user can only see "
                    "and approve plans submitted via the `plan_task` tool. "
                    "Take the same fields you just printed and pass them as "
                    "tool arguments to `plan_task`. Do not paste JSON in prose "
                    "again; call the tool now."
                )
            else:
                content = (
                    "[system] You are in plan mode. Stop explaining and call "
                    "`plan_task` now with problem, solution, and steps. "
                    "Use your best current understanding instead of waiting "
                    "for exhaustive research. `files_to_modify` and "
                    "`files_to_create` may be partial or empty if uncertain. "
                    "Do not respond with prose until the plan is approved."
                )
            self.messages.append({"role": "user", "content": content})
            return "continue"

        # Approved plan with unresolved steps — nudge to continue, but cap the
        # number of nudges so the agent doesn't loop forever after producing a
        # final answer (e.g., for an audit where steps are research-only).
        if (
            not is_bench
            and plan is not None
            and plan.approved
            and self.config.permission_mode != "plan"
            and plan.unresolved_steps()
            and st.unresolved_nudges < 3
        ):
            st.unresolved_nudges += 1
            if completion.text:
                self.messages.append(prose_msg(completion.text, completion.reasoning))
                if self.display:
                    self.display.flush_streaming_text()
            if st.plan_nudges < self.config.max_plan_nudges:
                st.plan_nudges += 1
                remaining = "; ".join(
                    f"{i + 1}. {step.description}"
                    for i, step in enumerate(plan.unresolved_steps()[:4])
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[system] You have an approved plan with unresolved steps. "
                            "Either call `update_plan` to mark each remaining step "
                            "(`done`/`skipped`/`blocked`) and continue, or call "
                            "`finish_plan` to resolve all remaining steps at once "
                            "and end the task. Do not just repeat the same prose. "
                            f"Remaining: {remaining}"
                        ),
                    }
                )
                return "continue"
            # Nudges exhausted — accept the prose answer and finish so we don't
            # loop indefinitely on research/audit tasks.
            st.prose_completions += 1
            self._sync_display_stats(st, turn)
            return self._build_result(
                st, success=True, final_text=completion.text, turn=turn,
            )

        # Empty response.
        if not (completion.text or "").strip():
            st.consecutive_errors += 1
            st.empty_responses += 1
            if st.consecutive_errors >= self.config.max_consecutive_errors:
                msg = "model produced empty responses"
                if self.display:
                    self.display.error(msg)
                return self._build_result(st, success=False, error=msg, turn=turn)
            if is_bench and st.consecutive_errors >= 3:
                self.messages.append({
                    "role": "user",
                    "content": (
                        "[system] CRITICAL: You have produced multiple empty responses. "
                        "You MUST act NOW. Either:\n"
                        "1. Call `read_file` on the file mentioned in the problem statement, OR\n"
                        "2. Call `edit_file` with your best fix attempt, OR\n"
                        "3. Respond with a plain text summary if you already fixed the bug.\n"
                        "Do NOT produce another empty response."
                    ),
                })
            else:
                self.messages.append({
                    "role": "user",
                    "content": (
                        "[system] Your last response was empty. You must either "
                        "call a tool or respond with a text summary of what you did. "
                        "Continue working on the task."
                    ),
                })
            return "continue"

        # Normal text-only completion — agent is done.
        st.prose_completions += 1
        if completion.text:
            self.messages.append(prose_msg(completion.text, completion.reasoning))
            if self.display:
                self.display.flush_streaming_text()
        self._sync_display_stats(st, turn)
        return self._build_result(st, success=True, final_text=completion.text, turn=turn)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self, start: float) -> TaskResult:
        st = LoopState(start=start)
        is_bench = self.config.permission_mode == "bench"
        _is_constrained = self.config.permission_mode in ("bench", "yolo")

        # Extract problem file paths and cache problem text before compaction.
        if _is_constrained:
            for msg in self.messages:
                if msg.get("role") == "user" and not str(msg.get("content", "")).startswith("[system]"):
                    st.problem_files = extract_problem_files(str(msg.get("content", "")))
                    break
            cache_problem_text(self, st)

        def _plan_active() -> bool:
            p = self.tool_ctx.plan
            return p is not None and p.approved

        _cached_perm_mode = self.config.permission_mode
        _cached_plan_active = _plan_active()
        _cached_schemas = openai_schemas(
            _cached_perm_mode, plan_active=_cached_plan_active,
        )

        for turn in range(1, self.config.max_turns + 1):
            # Finish countdown.
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
            now_plan_active = _plan_active()
            if (
                self.config.permission_mode != _cached_perm_mode
                or now_plan_active != _cached_plan_active
            ):
                _cached_perm_mode = self.config.permission_mode
                _cached_plan_active = now_plan_active
                _cached_schemas = openai_schemas(
                    _cached_perm_mode, plan_active=_cached_plan_active,
                )
                if self.display is not None:
                    self.display.set_mode(self.config.permission_mode)
            schemas = _cached_schemas

            self._refresh_system_injections()

            # Snapshot new messages to full_log before trim/compaction can destroy them.
            new_msgs = self.messages[self._full_log_idx:]
            if new_msgs:
                self._full_log.extend(new_msgs)
                self._full_log_idx = len(self.messages)

            # Compaction + trim.
            msg_count_before = len(self.messages)
            if getattr(self.client, "context_window", 0) > 0:
                self.messages[:] = await compact_messages(
                    self.messages, self.client,
                    context_limit=self.client.context_window,
                    threshold=self.config.compaction_threshold,
                )
            self.messages[:] = trim_history(
                self.messages, max_messages=self.config.max_history_messages,
            )
            self._last_persisted_idx = len(self.messages)
            self._full_log_idx = len(self.messages)

            if len(self.messages) < msg_count_before and self.tool_ctx.files_read_count:
                # Show relative paths so the model recognizes them.
                wd = self.tool_ctx.working_dir
                read_files = sorted(
                    os.path.relpath(p, wd) if os.path.isabs(p) else p
                    for p in self.tool_ctx.files_read_count.keys()
                )
                if len(read_files) > 15:
                    read_files = read_files[:15] + [f"... and {len(read_files) - 15} more"]
                inject_nudge(self, st, turn, (
                    f"[system] Context was compacted. Files you already read in this session: "
                    f"{', '.join(read_files)}. "
                    "If you need to edit a file but don't remember the exact content for "
                    "old_str, call read_file again to get the precise text — do NOT guess. "
                    "Use save_note to persist important content across compactions."
                ), force=True)

            # LLM call.
            try:
                completion = await self.client.complete(
                    self.messages, schemas, stream=True, on_text=self._on_text,
                )
            except LLMError as e:
                if self.display:
                    self.display.flush_streaming_text()
                    self.display.error(f"LLM error: {e}")
                return self._build_result(st, success=False, error=str(e), turn=turn - 1)

            # Always finalize the streaming display before any further console
            # output (tool headers, panels, prompts). Without this, subsequent
            # turns concatenate into the same Live buffer and the prior
            # narration is re-rendered on every refresh, producing repeated
            # text and muddying the approval prompt area.
            if self.display:
                self.display.flush_streaming_text()

            st.total_prompt_tokens += completion.prompt_tokens
            st.completion_tokens += completion.completion_tokens

            # --- No tool calls: prose-only completion ---
            if not completion.tool_calls:
                result = self._handle_prose_completion(completion, st, turn, is_bench)
                if isinstance(result, TaskResult):
                    return result
                continue

            self.messages.append(assistant_msg(completion.text, completion.tool_calls, completion.reasoning))

            # --- Compaction-resilient loop detection (all modes) ---
            # Build a key from all tool calls this turn and compare to previous.
            # In bench/yolo, threshold 7 so nudges (at 2) get a chance to work.
            # In plan/edits, threshold 5 so the user isn't kept waiting for
            # an obviously-stuck model.
            explore_blocked = False
            call_key = _call_key(completion.tool_calls)
            if call_key == st.last_call_key:
                st.consecutive_identical += 1
            else:
                st.consecutive_identical = 0
                st.last_call_key = call_key

            loop_threshold = 7 if _is_constrained else 5
            if st.consecutive_identical >= loop_threshold:
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
            # Mid-loop nudge — gentler in interactive modes.
            if not _is_constrained and st.consecutive_identical >= 2:
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[system] You repeated the same tool call "
                        f"{st.consecutive_identical + 1} times in a row with no "
                        "new information. Either try a different action, call "
                        "`finish_plan` if the work is done, or respond with a "
                        "plain-text summary to end the turn. Do not repeat "
                        "this call again."
                    ),
                })
            if _is_constrained and st.consecutive_identical >= 2:
                inject_consecutive_identical_nudge(self, st, turn=turn)

            # --- Dispatch tools ---
            plan_task_called_this_turn = False
            local_read_without_recall = 0
            dispatched_pairs: list[tuple[ToolCall, dict[str, Any]]] = []

            # Explore blocker (bench/yolo).
            _explore_eligible = _is_constrained and (
                (not st.files_edited
                 and st.turns_without_progress >= self.config.max_stuck_turns * 2)
                or (st.files_edited
                    and st.post_edit_read_turns >= self.config.max_post_edit_read_turns + 2)
            )
            _blocked_streak = (
                st.turns_without_progress - self.config.max_stuck_turns * 2
                if not st.files_edited
                else st.post_edit_read_turns - self.config.max_post_edit_read_turns - 2
            )
            explore_blocked = _explore_eligible and (_blocked_streak % 3 != 0)

            # Quality gate (always runs, even when explore_blocked).
            gate = apply_quality_gate(self, completion.tool_calls, st, turn)
            if isinstance(gate, TaskResult):
                return gate
            if gate == "skip":
                continue

            for tc in completion.tool_calls:
                if tc.name == "plan_task":
                    plan_task_called_this_turn = True
                st.total_tool_calls[tc.name] = st.total_tool_calls.get(tc.name, 0) + 1

                # Explore blocker.
                if explore_blocked:
                    hard_block = _blocked_streak >= self.config.max_stuck_turns * 2
                    blocked = False
                    if tc.name in EXPLORE_TOOLS:
                        if tc.name == "read_file":
                            path = str(tc.args.get("path", ""))
                            if path in st.recent_edit_fail_files or not hard_block and path not in self.tool_ctx.files_read:
                                blocked = False
                            else:
                                blocked = True
                        else:
                            blocked = True
                    elif tc.name == "run_command" and is_exploration_command(
                        str(tc.args.get("command", ""))
                    ):
                        blocked = True
                    if blocked:
                        hint = ""
                        if st.problem_files:
                            hint_files = [f for f in st.problem_files
                                          if f not in {str(p) for p in self.tool_ctx.files_read}]
                            if hint_files:
                                hint = (
                                    f" Try reading one of these files from the problem statement: "
                                    f"{', '.join(hint_files[:3])}."
                                )
                        append_tool_result(
                            self, tc,
                            message=(
                                '{"success": false, "error": "Exploration blocked: you have spent '
                                f'{st.turns_without_progress} turns reading/searching without making '
                                'any edits. You MUST call edit_file NOW with your best fix attempt. '
                                'Use the content you already have. A wrong fix that you iterate on '
                                f'is better than more exploration.{hint}"}}'
                            ),
                        )
                        continue

                outcome = await run_tool(self, turn, tc)
                dispatched_pairs.append((tc, outcome))

                # Safety checks.
                inject_test_failure_nudge(self, st, tc, outcome, turn=turn)
                check_goal_drift(self, st, tc, outcome, turn=turn)
                track_edit_failure(self, st, tc, outcome, turn=turn)
                track_tool_outcome(self, st, tc, outcome)

                if tc.name in ("read_file", "list_directory", "search_files") and outcome["success"]:
                    local_read_without_recall += 1

                # Plan-approved terminal event.
                if outcome.get("plan_approved") and self.config.permission_mode == "plan":
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

            # Recall-first enforcement (plan mode only) — outside per-tool loop
            # to avoid triangular accumulation of local_read_without_recall.
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
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "[system] Too many read calls without `recall`. Call `recall(query=...)` now "
                            "to find relevant files, or call `plan_task` if you have enough information. "
                            "Do not call read_file, list_directory, or search_files again until you use recall."
                        ),
                    })
                    self.consecutive_reads_without_recall = 0
                    continue

            # Reset identical-call counter when an edit succeeded this turn.
            if _is_constrained and any(
                tc.name == "edit_file" and outcome.get("success")
                for tc, outcome in dispatched_pairs
            ):
                st.consecutive_identical = 0
                st.last_call_key = ""

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
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    "[system] You have investigated enough. Stop calling "
                                    "read tools and call `plan_task` now. Use the evidence "
                                    "you already have instead of waiting for exhaustive "
                                    "research. `files_to_modify` and `files_to_create` may "
                                    "be partial or empty if uncertain. Do not read any more "
                                    "files before calling `plan_task`."
                                ),
                            })
                            continue
                        else:
                            msg = "plan-mode run finished without producing a plan_task"
                            if self.display:
                                self.display.error(msg)
                            return self._build_result(st, success=False, error=msg, turn=turn)

            # Phase tracking (bench/yolo).
            phase_result = update_phase(self, st, dispatched_pairs, turn=turn)
            if isinstance(phase_result, TaskResult):
                return phase_result

            # Force-finish after test pass.
            if _is_constrained and st.test_passed_after_edit:
                st.test_passed_after_edit = False
                st.finish_countdown = 2
                self.messages.append({
                    "role": "user",
                    "content": (
                        "[system] A test/verification command passed after your edits. "
                        "Your fix is working. You are DONE. Respond with ONLY a plain text "
                        "summary of what you changed and why. Do NOT call any more tools. "
                        "Do NOT run more commands. Just write text and stop."
                    ),
                })

            # Read-only spiral detection.
            spiral = check_read_only_spiral(self, st, is_bench, turn)
            if spiral is not None:
                return spiral

            # Turn budget + re-anchoring + stuck detection.
            inject_turn_budget(self, st, turn)
            maybe_reanchor_problem(self, st, turn)
            apply_stuck_detection(self, st, is_bench, turn=turn)

            # Per-turn event log (bench/yolo only).
            if _is_constrained:
                tools_this_turn = [
                    {"name": tc.name, "args_summary": brief(tc) or str(tc.args.get("command", ""))[:80]}
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
        """Handle plan_task approval flow. Returns (outcome, plan_approved).

        ``prompt_fn`` may return:
          - True / False: approve or decline.
          - ``("feedback", "<text>")``: declined, but pass the user's feedback
            back to the model so it can revise the plan.
        """
        if self.display:
            self.display.plan_panel(outcome.data)
        reply: Any = True
        feedback: str = ""
        if self.prompt_fn is not None:
            from squishy.tools.base import Tool
            try:
                reply = await self.prompt_fn(
                    Tool(name="plan_task", description="", parameters={},
                         run=lambda *_: None),  # type: ignore[arg-type]
                    tc.args,
                )
            except EOFError:
                reply = False
            except KeyboardInterrupt:
                # User wants to abort the whole turn, not just decline
                # this plan. Drop the persisted plan so a future run
                # starts clean, then propagate so Agent.run translates
                # this into AgentCancelled.
                self.tool_ctx.plan = None
                self.tool_ctx.pending_plan_evidence.clear()
                self.tool_ctx.plan_switch_prompted = False
                clear_plan(self.tool_ctx.working_dir)
                raise

        if isinstance(reply, tuple) and len(reply) == 2 and reply[0] == "feedback":
            approved = False
            feedback = str(reply[1] or "").strip()
        else:
            approved = bool(reply)

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
        if feedback:
            err = f"Plan declined. User feedback: {feedback}"
            if self.display:
                self.display.info(f"[plan] feedback: {feedback}")
        else:
            err = "Plan declined by user. Ask for changes or a new approach."
        return ToolResult(False, error=err), False

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
            self.display.turn_header(
                turn, self.config.max_turns, tc.name, brief,
                mode=self.config.permission_mode,
            )

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
            elif outcome.success and tc.name in ("update_plan", "finish_plan"):
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


def _looks_like_json_plan(text: str) -> bool:
    """Heuristic: did the model write a plan_task JSON literal in prose?

    The signature we care about is the simultaneous presence of the three
    plan_task field names quoted as JSON keys. We don't try to parse —
    just to detect the failure mode and route to a sharper nudge.
    """
    if not text or "{" not in text:
        return False
    needles = ('"problem"', '"solution"', '"steps"')
    return all(n in text for n in needles)


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
