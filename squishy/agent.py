"""Async agent loop — slim orchestrator.

Delegates to:
  agent_state.py     — TaskResult, LoopState, message helpers
  agent_safety.py    — quality gates, informational nudges
  agent_dispatch.py  — tool dispatch, plan approval, evidence
  agent_phases.py    — turn budget, re-anchoring, problem caching
  phase_machine.py   — phase-gated tool availability (bench mode)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

from squishy.agent_dispatch import run_tool, track_tool_outcome
from squishy.agent_phases import (
    cache_problem_text,
    inject_turn_budget,
    maybe_post_edit_pytest_nudge,
    maybe_reanchor_problem,
)
from squishy.agent_safety import (
    apply_quality_gate,
    check_goal_drift,
    detect_shell_file_read,
    inject_nudge,
    inject_test_failure_nudge,
    needs_f2p_verification,
    track_edit_failure,
)
from squishy.agent_state import (
    LoopState,
    TaskResult,
    assistant_msg,
    brief,
    call_key,
    extract_problem_files,
    prose_msg,
)
from squishy.phase_machine import PhaseState, advance, check_finish_plan_gate, check_transition
from squishy.client import Client, CompletionResult, ToolCall
from squishy.config import Config
from squishy.context import (
    build_system_prompt,
    compact_messages,
    detect_project,
    normalize_messages,
    trim_history,
)
from squishy.display import Display, estimate_tokens
from squishy.errors import AgentCancelled, AgentTimeout, LLMError
from squishy.index.store import has_index
from squishy.plan_state import load_plan, render_plan_status
from squishy.tools import PromptFn, ToolContext, openai_schemas
from squishy.tools.scratchpad import render_notes

log = logging.getLogger("squishy.agent")


# F3: heuristics for "the agent observed a test failure but did not edit
# anything afterwards."  Strings/regexes are deliberately broad — false
# positives waste at most one extra turn (the gate is single-shot), false
# negatives let buggy patches ship.
_TEST_FAIL_RE = re.compile(
    r"(?ix)"
    r"AssertionError"
    r"|\bFAILED\b"
    r"|\bERROR\b\s+(?:tests?/|test_)"
    r"|\b\d+\s+failed\b"
    r"|\b\d+\s+errors?\b"
    r"|test\s+pass(?:ed)?\s*[:=]\s*false"
    r"|\"failed\"\s*:\s*[1-9]"
    r"|\"errors\"\s*:\s*[1-9]"
)


def _has_unaddressed_test_failure(
    messages: list[dict[str, Any]], lookback: int = 12,
) -> str | None:
    """Return a short summary if the agent recently observed a test failure
    via ``run_command`` and has NOT edited any file since.  Otherwise None.

    Walks the trailing ``lookback`` messages; ignores anything older.
    """
    if not messages:
        return None
    tail = messages[-lookback:]
    failure_summary: str | None = None
    failure_idx: int = -1
    for i, m in enumerate(tail):
        if m.get("role") != "tool":
            continue
        if m.get("name") != "run_command":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        match = _TEST_FAIL_RE.search(content)
        if match:
            # Most recent failure wins.
            failure_summary = match.group(0).strip()
            failure_idx = i
    if failure_summary is None or failure_idx < 0:
        return None
    # Did any edit_file/write_file land AFTER the failure was observed?
    for j in range(failure_idx + 1, len(tail)):
        m = tail[j]
        if m.get("role") != "tool":
            continue
        if m.get("name") in ("edit_file", "write_file"):
            return None
    # Truncate the summary so the nudge stays terse.
    return failure_summary[:120]


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
    # Active loop state — published by _run_loop so the outer run() handler
    # can build a partial TaskResult on timeout/cancellation instead of
    # discarding the in-progress transcript and turn_log.
    _active_st: LoopState | None = field(init=False, default=None)
    _active_turn: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.tool_ctx = ToolContext(
            working_dir=self.config.working_dir,
            permission_mode=self.config.permission_mode,
            sandbox_image=self.config.sandbox_image,
            use_sandbox=self.config.use_sandbox,
            plan=load_plan(self.config.working_dir),
            max_tool_output_chars=self.config.max_tool_output_chars,
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
            err = AgentTimeout(f"task exceeded {timeout}s")
            # Attach a partial TaskResult so the bench harness can recover
            # the transcript / turn_log accumulated up to the timeout.
            # Without this, AgentTimeout discards full_log and the agent's
            # last ~50 turns disappear from diagnostics — which is exactly
            # what hid the v27p3 scico-561 prompt-build behavior from us.
            partial = self._build_partial_result(
                error=f"AgentTimeout: task exceeded {timeout}s",
            )
            if partial is not None:
                err.partial_result = partial
            raise err from e
        except asyncio.CancelledError:
            err = AgentCancelled("task cancelled by caller")
            partial = self._build_partial_result(error="cancelled")
            if partial is not None:
                err.partial_result = partial
            raise err from None
        except KeyboardInterrupt:
            # Ctrl+C anywhere inside the run — including inside an approval
            # prompt — should abort the whole turn, not just decline the
            # current tool. Translate into our usual cancelled signal so
            # the CLI's outer handler resets cleanly.
            if self.display:
                self.display.stop_thinking()
                self.display.flush_streaming_text()
            raise AgentCancelled("interrupted by user") from None
        finally:
            # Belt-and-suspenders: a spinner left running across a
            # CancelledError / TimeoutError path will keep refreshing
            # into the REPL prompt area until GC fires.
            if self.display is not None:
                self.display.stop_thinking()

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

    def _build_partial_result(self, *, error: str) -> TaskResult | None:
        """Build a TaskResult from in-progress LoopState (for timeout/cancel
        paths).  Returns None if the loop never started.
        """
        st = self._active_st
        if st is None:
            return None
        try:
            return self._build_result(
                st, success=False, error=error, turn=self._active_turn,
            )
        except Exception:  # noqa: BLE001
            log.warning("partial result build failed", exc_info=True)
            return None

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
        # Filter out the transient live-context pair — it's a view-only
        # artifact that must never reach the session log.
        new = [m for m in new if not m.get(self._LIVE_CTX_MARKER)]
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
    # Live-context injection (cache-stable system prefix)
    # ------------------------------------------------------------------
    #
    # Plan-status and notes change between turns but we never want to touch
    # ``messages[0]`` after init — every byte change there evicts the vLLM
    # prefix cache (the most expensive thing the server does each turn).
    # Instead we strip+rebuild a synthetic ``(assistant tool_calls, tool
    # result)`` pair at the tail of ``self.messages`` each turn. Both
    # messages are tagged with ``_LIVE_CTX_MARKER`` so they can be reliably
    # identified, filtered out of persistence/full-log snapshots, and
    # removed before the next rebuild.
    #
    # The pair MUST be well-formed (matching ``tool_call_id``) because
    # ``_strip_orphan_assistant_tool_calls`` in ``context.py`` aggressively
    # removes orphan tool messages (Azure-strict requirement).

    _LIVE_CTX_MARKER = "_squishy_live_ctx"
    _LIVE_CTX_TOOL_NAME = "_squishy_context"
    _LIVE_CTX_CALL_ID = "squishy-live-ctx"

    def _strip_live_context_pair(self) -> None:
        """Remove any previously-injected live-context messages."""
        if not self.messages:
            return
        self.messages[:] = [
            m for m in self.messages if not m.get(self._LIVE_CTX_MARKER)
        ]

    def _refresh_live_context_pair(self) -> None:
        """Strip any prior live-context pair and append a fresh one.

        Built from ``tool_ctx.plan`` (via ``render_plan_status``) and
        ``tool_ctx.notes`` (via ``render_notes``). Returns without
        appending when both are empty — the prior pair has already been
        stripped, so the message list is back to canonical state.

        Must be called AFTER ``trim_history``/``compact_messages`` so the
        pair never participates in those routines.
        """
        self._strip_live_context_pair()
        plan = self.tool_ctx.plan
        parts: list[str] = []
        if plan is not None:
            parts.append(render_plan_status(plan))
        if self.tool_ctx.notes:
            parts.append(render_notes(self.tool_ctx.notes))
        if not parts:
            return
        content = "\n\n".join(parts)
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": self._LIVE_CTX_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": self._LIVE_CTX_TOOL_NAME,
                        "arguments": "{}",
                    },
                }
            ],
            self._LIVE_CTX_MARKER: True,
        }
        tool_result = {
            "role": "tool",
            "tool_call_id": self._LIVE_CTX_CALL_ID,
            "name": self._LIVE_CTX_TOOL_NAME,
            "content": content,
            self._LIVE_CTX_MARKER: True,
        }
        self.messages.append(assistant)
        self.messages.append(tool_result)

    # ------------------------------------------------------------------
    # v2 auto-pytest finish gate
    # ------------------------------------------------------------------

    async def _auto_run_f2p_pytest(self, st: LoopState, turn: int) -> bool:
        """Synthesize a run_command tool call that runs the F2P tests.

        Used by the v2 finish gate when the agent tries to wrap up without
        having actually run the failing tests since its last edit. The
        result lands in ``self.messages`` naturally via ``run_tool``, so
        the agent sees it as its own tool result on the next turn.

        Returns True if a pytest run was actually dispatched.
        """
        if not st.fail_to_pass_tests:
            return False
        nodeids = list(st.fail_to_pass_tests)[:5]
        cmd = (
            "python -m pytest --tb=short --no-header -p no:cacheprovider "
            + " ".join(shlex.quote(n) for n in nodeids)
        )
        tc = ToolCall(
            id=f"auto-pytest-{st.auto_pytest_runs}",
            name="run_command",
            args={"command": cmd, "timeout": 120},
        )
        # Append the paired assistant tool_calls message BEFORE dispatching, so
        # the synthetic run_command result isn't a reverse-orphan tool message
        # (strict endpoints 400 on an unpaired role="tool"). run_tool only
        # appends the tool result; it does not synthesize the assistant call.
        self.messages.append(assistant_msg("", [tc]))
        await run_tool(self, turn, tc)
        st.auto_pytest_runs += 1
        return True

    # ------------------------------------------------------------------
    # Prose completion handling
    # ------------------------------------------------------------------

    async def _handle_prose_completion(
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
        # F3: in bench/yolo, intercept once if the agent observed a test
        # failure via run_command but never edited anything to fix it.
        # This catches the v27 regression where the model ran a repro
        # script, saw "Adjoint test pass: False", and shipped the patch
        # anyway.  Single-shot so genuinely unfixable runs can terminate.
        if (
            self.config.permission_mode in ("bench", "yolo")
            and st.no_progress_intercepts < 1
        ):
            failure = _has_unaddressed_test_failure(self.messages)
            if failure:
                # Persist the prose first so the nudge has the agent's
                # last words for grounding.
                if completion.text:
                    self.messages.append(
                        prose_msg(completion.text, completion.reasoning)
                    )
                    if self.display:
                        self.display.flush_streaming_text()
                st.no_progress_intercepts += 1
                injected = inject_nudge(
                    self, st, turn,
                    "[system] You observed a test failure "
                    f"({failure!r}) but did not call `edit_file` or "
                    "`write_file` after it.  Either fix the underlying "
                    "bug now with `edit_file`, or call "
                    "`finish_plan(status=\"failure\")` if the bug is "
                    "genuinely unfixable.",
                    min_gap=0, force=True,
                )
                if injected:
                    return "continue"

        # v2 auto-pytest finish gate (Site A — natural finish).  In bench
        # mode, if the agent edited but never ran F2P pytest since the
        # last edit, synthesize a pytest run and nudge the model to react.
        # Capped by max_auto_pytest_runs so this can't loop indefinitely.
        if (
            self.config.permission_mode == "bench"
            and needs_f2p_verification(st)
            and st.auto_pytest_runs < self.config.max_auto_pytest_runs
        ):
            if completion.text:
                self.messages.append(
                    prose_msg(completion.text, completion.reasoning)
                )
                if self.display:
                    self.display.flush_streaming_text()
            ran = await self._auto_run_f2p_pytest(st, turn)
            if ran:
                inject_nudge(
                    self, st, turn,
                    "[system] You tried to finish without running the failing "
                    "tests. I ran them for you — see the run_command tool "
                    "result above. Address what you see: either edit_file / "
                    "write_file to fix what the test reports, or call "
                    "finish_plan(status=\"failure\") if it is genuinely "
                    "unfixable.",
                    min_gap=0, force=True,
                )
                return "continue"

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
        # Publish so run()'s timeout/cancel handlers can recover state.
        self._active_st = st
        self._active_turn = 0
        is_bench = self.config.permission_mode == "bench"
        _is_constrained = self.config.permission_mode in ("bench", "yolo")

        # Extract problem file paths and cache problem text before compaction.
        if _is_constrained:
            for msg in self.messages:
                if msg.get("role") == "user" and not str(msg.get("content", "")).startswith("[system]"):
                    st.problem_files = extract_problem_files(str(msg.get("content", "")))
                    break
            cache_problem_text(self, st)

        # Phase machine (bench mode only — interactive modes are unaffected).
        ps: PhaseState | None = None
        if is_bench:
            ps = PhaseState(
                max_explore_turns=self.config.max_explore_turns,
                max_plan_turns=self.config.max_plan_turns,
                max_fix_verify_cycles=self.config.max_fix_verify_cycles,
                # F5: surface FAIL_TO_PASS to the phase machine so its
                # check_transition can require coverage of every distinct
                # F2P file before flipping test_passed_after_edit True.
                fail_to_pass=list(st.fail_to_pass_tests),
            )

        def _plan_active() -> bool:
            p = self.tool_ctx.plan
            return p is not None and p.approved

        _cached_perm_mode = self.config.permission_mode
        _cached_plan_active = _plan_active()
        _cached_phase = ps.phase if ps else None
        _cached_schemas = openai_schemas(
            _cached_perm_mode, plan_active=_cached_plan_active,
            phase=_cached_phase,
        )

        for turn in range(1, self.config.max_turns + 1):
            self._active_turn = turn
            # Done phase (bench): no tools available, model must produce prose.
            if ps and ps.phase == "done":
                # v2 auto-pytest finish gate (Site B — done phase).  Same
                # logic as Site A: edited but no F2P pytest since.  If the
                # gate fires, kick the phase machine back to execute and
                # let the agent react.
                if (
                    self.config.permission_mode == "bench"
                    and needs_f2p_verification(st)
                    and st.auto_pytest_runs < self.config.max_auto_pytest_runs
                ):
                    ran = await self._auto_run_f2p_pytest(st, turn)
                    if ran:
                        inject_nudge(
                            self, st, turn,
                            "[system] You tried to finish without running the "
                            "failing tests. I ran them for you — see the "
                            "run_command tool result above. Address what you "
                            "see: either edit_file / write_file to fix what "
                            "the test reports, or call "
                            "finish_plan(status=\"failure\") if it is "
                            "genuinely unfixable.",
                            min_gap=0, force=True,
                        )
                        ps.phase = "execute"
                        continue
                if self.display:
                    self.display.warn("done phase — force finishing")
                return self._build_result(
                    st, success=True,
                    final_text="Fix applied and verified.",
                    turn=turn,
                )

            self.tool_ctx.permission_mode = self.config.permission_mode
            now_plan_active = _plan_active()
            now_phase = ps.phase if ps else None
            if (
                self.config.permission_mode != _cached_perm_mode
                or now_plan_active != _cached_plan_active
                or now_phase != _cached_phase
            ):
                _cached_perm_mode = self.config.permission_mode
                _cached_plan_active = now_plan_active
                _cached_phase = now_phase
                _cached_schemas = openai_schemas(
                    _cached_perm_mode, plan_active=_cached_plan_active,
                    phase=_cached_phase,
                )
                if self.display is not None:
                    self.display.set_mode(self.config.permission_mode)
            schemas = _cached_schemas

            # Strip any prior live-context pair so it never appears in
            # full_log snapshots, persistence, trim_history, or
            # compact_messages. It will be rebuilt below after trim/compact.
            self._strip_live_context_pair()

            # Snapshot new messages to full_log before trim/compaction can destroy them.
            new_msgs = self.messages[self._full_log_idx:]
            if new_msgs:
                self._full_log.extend(new_msgs)
                self._full_log_idx = len(self.messages)

            # Compaction + trim.
            msg_count_before = len(self.messages)
            did_compact = False
            if getattr(self.client, "context_window", 0) > 0:
                compacted_msgs = await compact_messages(
                    self.messages, self.client,
                    context_limit=self.client.context_window,
                    threshold=self.config.compaction_threshold,
                )
                if len(compacted_msgs) < len(self.messages):
                    did_compact = True
                self.messages[:] = compacted_msgs
            # Dynamic history sizing: scale with the model's context window
            # so 128k models keep more history than 32k models.  Bounded so
            # we don't blow up on absurdly long contexts.  Compaction at
            # 70% remains the second safety valve.
            ctx = getattr(self.client, "context_window", 0) or 0
            if ctx > 0:
                dyn_max = max(10, min(60, ctx // 4096))
                hist_cap = max(self.config.max_history_messages, dyn_max)
            else:
                hist_cap = self.config.max_history_messages
            self.messages[:] = trim_history(
                self.messages, max_messages=hist_cap,
            )
            self._last_persisted_idx = len(self.messages)
            self._full_log_idx = len(self.messages)

            # Rebuild the live-context pair AFTER trim/compact. The pair
            # carries plan-status + notes for the model but is kept out of
            # the canonical history so the system prefix stays byte-stable
            # turn-over-turn (vLLM prefix cache stays warm).
            self._refresh_live_context_pair()

            if did_compact:
                st.compaction_count += 1

                # Force-finish if too many compactions without any edits.
                if _is_constrained and st.compaction_count >= 5 and not st.files_edited:
                    msg = (
                        f"force finishing: {st.compaction_count} context compactions "
                        "without any file edits"
                    )
                    if self.display:
                        self.display.warn(msg)
                    return self._build_result(st, success=False, error=msg, turn=turn)

            # Compaction reminder (informational — lists already-read files).
            if did_compact and self.tool_ctx.files_read_count:
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
                ), min_gap=5, force=True)

            # Re-inject index-recall pointers after compaction (bench only).
            # The original "## Relevant Code (from index)" section lives in
            # the first user message; once compaction summarises it the
            # model loses the breadcrumb trail to where the bug likely
            # lives.  Re-surface a compact pointer list so the model can
            # navigate without re-running recall by hand.  Pointers only
            # — no file bodies — to keep the nudge cheap.
            if did_compact and is_bench and st.problem_text:
                try:
                    from squishy.bench.swebench import _recall_from_index
                    pointers = _recall_from_index(
                        self.config.working_dir, st.problem_text, limit=5,
                    )
                except Exception:  # noqa: BLE001
                    pointers = []
                if pointers:
                    lines = ["[system] Post-compaction recall — likely-relevant code:"]
                    for p in pointers:
                        line_info = ""
                        if p.get("lines"):
                            line_info = f" (L{p['lines'][0]}-{p['lines'][1]})"
                        lines.append(
                            f"- `{p['path']}`{line_info}: "
                            f"{p.get('kind', 'file')} `{p['name']}`"
                        )
                    lines.append(
                        "Use `read_file` to re-load whichever of these you "
                        "need; do not guess at file contents from memory."
                    )
                    inject_nudge(
                        self, st, turn, "\n".join(lines),
                        min_gap=5, force=True,
                    )

            # Enforce the assistant↔tool pairing invariant on the exact list
            # we send. Any nudge/gate/compaction that severed a pair or
            # injected a stray tool result is repaired here, once, so no
            # malformed transcript reaches the endpoint.
            self.messages[:] = normalize_messages(self.messages)

            # LLM call. In interactive modes, show a spinner so a slow
            # first token / cold model doesn't look like a hung process.
            # The spinner cancels itself on the first streamed chunk.
            if self.display is not None and not _is_constrained:
                self.display.start_thinking()
            try:
                completion = await self.client.complete(
                    self.messages, schemas, stream=True, on_text=self._on_text,
                    on_retry=self._on_client_retry,
                )
            except LLMError as e:
                if self.display:
                    self.display.flush_streaming_text()
                    self.display.error(f"LLM error: {e}")
                # Content-filter rejections and Azure-strict schema violations
                # (orphan tool_calls) are deterministic — no point retrying.
                # Fail-fast so the bench harness moves to the next instance
                # instead of burning the per-task budget on a guaranteed-fail prompt.
                err_str = str(e).lower()
                if (
                    "content_filter" in err_str
                    or "azure_strict_orphan_tool_calls" in err_str
                ):
                    log.warning("deterministic LLM error — aborting instance: %s", e)
                    return self._build_result(st, success=False, error=str(e), turn=turn - 1)
                if is_bench:
                    st.llm_errors += 1
                    if st.llm_errors >= 3:
                        return self._build_result(st, success=False, error=str(e), turn=turn - 1)
                    log.warning("LLM error in bench mode (attempt %d/3), retrying: %s",
                                st.llm_errors, e)
                    continue
                return self._build_result(st, success=False, error=str(e), turn=turn - 1)

            if self.display:
                self.display.flush_streaming_text()

            # Retry-storm short-circuit: tenacity may have eaten dozens of
            # seconds inside the call.  Accumulate the per-call retry count
            # and escalate when upstream is clearly unstable so we don't burn
            # the whole task_timeout in silence (v25 gemma-1 lost ~900s here).
            call_retries = getattr(self.client, "last_call_retries", 0)
            if call_retries:
                st.cumulative_retries += call_retries
                if is_bench:
                    if st.cumulative_retries >= 24:
                        # Hard ceiling: bail to capture-on-error path so the
                        # workspace diff (any partial edits) still survives.
                        raise AgentTimeout(
                            f"retry storm: {st.cumulative_retries} cumulative "
                            f"upstream retries — aborting to preserve partial work"
                        )
                    if st.cumulative_retries >= 12:
                        inject_nudge(self, st, turn, (
                            f"[system] CRITICAL: the upstream LLM API has been "
                            f"unstable ({st.cumulative_retries} cumulative retries). "
                            "Wrap up immediately — submit your best current edit "
                            "and stop. Do NOT call additional tools unless you have "
                            "no patch yet."
                        ), min_gap=0, force=True)

            st.total_prompt_tokens += completion.prompt_tokens
            st.completion_tokens += completion.completion_tokens

            # --- No tool calls: prose-only completion ---
            if not completion.tool_calls:
                result = await self._handle_prose_completion(completion, st, turn, is_bench)
                if isinstance(result, TaskResult):
                    return result
                continue

            self.messages.append(assistant_msg(completion.text, completion.tool_calls, completion.reasoning))

            # --- Loop detection (all modes) ---
            current_key = call_key(completion.tool_calls)
            if current_key == st.last_call_key:
                st.consecutive_identical += 1
            else:
                st.consecutive_identical = 0
                st.last_call_key = current_key

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
            # Mid-loop nudge (interactive modes only). Deferred to AFTER the
            # tool results are appended — inserting a user message between the
            # assistant tool_calls and its results violates the API pairing
            # invariant (assistant tool_calls must be immediately followed by
            # tool messages). Captured here, appended after dispatch.
            pending_repeat_nudge: str | None = None
            if not _is_constrained and st.consecutive_identical >= 2:
                pending_repeat_nudge = (
                    f"[system] You repeated the same tool call "
                    f"{st.consecutive_identical + 1} times in a row with no "
                    "new information. Either try a different action, call "
                    "`finish_plan` if the work is done, or respond with a "
                    "plain-text summary to end the turn. Do not repeat "
                    "this call again."
                )

            # --- Quality gate ---
            gate = apply_quality_gate(self, completion.tool_calls, st, turn)
            if isinstance(gate, TaskResult):
                return gate
            if gate == "skip":
                continue

            # --- Dispatch tools ---
            plan_task_called_this_turn = False
            local_read_without_recall = 0
            dispatched_pairs: list[tuple[ToolCall, dict[str, Any]]] = []

            for tc in completion.tool_calls:
                if tc.name == "plan_task":
                    plan_task_called_this_turn = True
                st.total_tool_calls[tc.name] = st.total_tool_calls.get(tc.name, 0) + 1

                outcome = await run_tool(self, turn, tc)
                dispatched_pairs.append((tc, outcome))

                # Informational feedback.
                inject_test_failure_nudge(self, st, tc, outcome, turn=turn)
                check_goal_drift(self, st, tc, outcome, turn=turn)
                edit_loop_result = track_edit_failure(self, st, tc, outcome, turn=turn)
                if edit_loop_result is not None:
                    self._sync_display_stats(st, turn)
                    return edit_loop_result
                detect_shell_file_read(self, st, tc, outcome, turn=turn)
                track_tool_outcome(self, st, tc, outcome, turn=turn)

                if tc.name in ("read_file", "list_directory", "search_files") and outcome["success"]:
                    local_read_without_recall += 1

                # Plan-approved terminal event (interactive plan mode).
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

            # Deferred repeat-nudge: now that all tool results are appended,
            # it's safe to add the trailing user message (see capture above).
            if pending_repeat_nudge is not None:
                self.messages.append({"role": "user", "content": pending_repeat_nudge})

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

            # Plan-mode investigation nudge (interactive only).
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

            # --- Phase transitions (bench mode) ---
            if ps:
                # F2P finish-plan gate: block finish_plan when the agent
                # claims done without having actually passed the FAIL_TO_PASS
                # tests.  Releases after one intercept so a degraded test env
                # cannot trap the agent forever.
                gate_msg = check_finish_plan_gate(
                    ps, dispatched_pairs,
                    st.fail_to_pass_tests,
                    st.f2p_finish_gate_intercepts,
                    max_intercepts=self.config.max_finish_gate_intercepts,
                    last_f2p_failures=st.last_f2p_failures,
                    last_f2p_collection_error=st.last_f2p_collection_error,
                )
                if gate_msg:
                    st.f2p_finish_gate_intercepts += 1
                    inject_nudge(self, st, turn, gate_msg, min_gap=0, force=True)
                    # Do NOT transition to done — skip check_transition this turn.
                    # The agent will get another shot to run the right tests.
                    continue
                transition = check_transition(ps, dispatched_pairs)
                # Always sync diagnostics from phase machine to LoopState.
                st.phase = ps.phase
                st.explore_turns = ps.explore_turns
                st.fix_verify_cycles = ps.fix_verify_cycles
                st.test_passed_after_edit = ps.test_passed_after_edit
                # F5: keep LoopState's coverage view in sync with the phase
                # machine's authoritative tally for the diagnostics export.
                st.f2p_files_covered = set(ps.f2p_files_covered)
                if transition.force_finish:
                    if self.display:
                        self.display.warn(
                            f"phase: exhausted fix-verify budget ({ps.fix_verify_cycles} cycles)"
                        )
                    return self._build_result(
                        st,
                        success=transition.force_finish_success,
                        final_text="Fix applied. Agent exhausted edit-verify cycle budget." if ps.has_edit else "",
                        turn=turn,
                    )
                if transition.new_phase:
                    advance(ps, transition)
                    # Re-sync after advance (phase changed).
                    st.phase = ps.phase
                # Inject informational notification (phase change or in-phase nudge).
                if transition.notification:
                    inject_nudge(self, st, turn, transition.notification, min_gap=2)

            # Turn budget + re-anchoring (informational).
            inject_turn_budget(self, st, turn)
            maybe_reanchor_problem(self, st, turn)
            maybe_post_edit_pytest_nudge(self, st, turn)

            # Per-turn event log (bench/yolo only).
            if _is_constrained:
                tools_this_turn = [
                    {"name": tc.name, "args_summary": brief(tc) or str(tc.args.get("command", ""))[:80]}
                    for tc in completion.tool_calls
                ]
                st.turn_log.append({
                    "turn": turn,
                    "phase": ps.phase if ps else st.phase,
                    "tools": tools_this_turn,
                    "dispatched": len(dispatched_pairs),
                    "blocked": len(completion.tool_calls) - len(dispatched_pairs),
                    "consecutive_identical": st.consecutive_identical,
                    "quality_violations": st.total_quality_violations,
                    "files_edited": len(st.files_edited),
                    "fix_verify_cycles": ps.fix_verify_cycles if ps else st.fix_verify_cycles,
                    "explore_turns": ps.explore_turns if ps else st.explore_turns,
                    "elapsed_s": round(time.monotonic() - st.start, 1),
                })

            self._persist_new_messages()

        # Max turns exhausted.
        msg = f"max turns ({self.config.max_turns}) reached"
        if self.display:
            self.display.warn(msg)
            self._sync_display_stats(st, self.config.max_turns)
        return self._build_result(st, success=False, error=msg, turn=self.config.max_turns)

    async def _on_text(self, chunk: str) -> None:
        if self.display:
            self.display.streaming_text_chunk(chunk)

    def _on_client_retry(
        self, attempt: int, max_attempts: int, exc: BaseException,
    ) -> None:
        """Called by Client when a transient failure forces a retry.

        Two jobs:
        1. Drop the partial in-flight stream — without ``reset_streaming``
           the next attempt's chunks would concatenate with what the user
           already saw, producing garbled markdown.
        2. Surface the retry inline so a multi-second tenacity backoff
           doesn't look like a hung process. Bench mode skips the
           print since there's no human watching.
        """
        if self.display is None:
            return
        self.display.reset_streaming()
        if self.config.permission_mode in ("bench",):
            return
        # Compact summary — exception class is usually enough for the
        # user to know whether it's network or server-side.
        ex_name = type(exc).__name__ if exc is not None else "transient error"
        self.display.warn(
            f"upstream {ex_name}; retrying ({attempt}/{max_attempts})…"
        )
        # Restart the spinner so the wait between retries isn't silent.
        self.display.start_thinking(label="retrying")


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
