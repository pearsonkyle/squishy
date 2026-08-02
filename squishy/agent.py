"""Async agent loop — slim orchestrator.

Delegates to:
  agent_state.py     — TaskResult, LoopState, message helpers
  agent_safety.py    — the nudge primitive
  agent_dispatch.py  — tool dispatch and outcome tracking

The loop deliberately does very little between the model and its tools:
complete, dispatch, append, repeat. Everything it *used* to do in between —
a quality gate, a phase machine, a plan protocol, goal-drift and edit-failure
detectors, turn budgets, re-anchoring — was removed after measurement showed
harness-generated refusals were the largest single failure bucket on
SWE-rebench, and that the quality gate's "skip this turn" path erased the
model's own tool call from its history (see agent_safety's module docstring).

Loop-breaking now lives in the tools, where the response can ride back in the
tool result instead of arriving as an out-of-band user message.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from squishy.agent_dispatch import run_tool, track_tool_outcome
from squishy.agent_safety import inject_nudge
from squishy.agent_state import (
    LoopState,
    TaskResult,
    assistant_msg,
    brief,
    call_key,
    extract_problem_files,
    looks_like_file_write,
    prose_msg,
)
from squishy.client import Client, CompletionResult
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
from squishy.graph import has_graph
from squishy.index.store import has_index
from squishy.tool_aliases import normalize_call
from squishy.tools import PromptFn, ToolContext, openai_schemas
from squishy.tools.scratchpad import render_notes

log = logging.getLogger("squishy.agent")

# How often to remind a model that has not yet changed a file. This is the one
# piece of edit pressure that survived, and it is a nudge rather than a gate:
# the blocking version withdrew `run_command`, and on aiohttp the model
# answered all 25 refusals by calling it again — 25 turns burned, no patch,
# while the ungated arm patched. Without any pressure at all, a shell run
# explored for 80 turns and also produced nothing, so the nudge stays.
# What to say to a model that stopped without editing anything. Deliberately
# short and imperative: it is read at the moment the model believes it is done.
EMPTY_PATCH_NUDGE = (
    "[system] You stopped, but no file has been changed — there is nothing to "
    "grade, so this run currently scores zero. Do not summarize again. Edit "
    "the non-test source file your analysis points at, right now, using the "
    "best explanation you have. A partial fix scores more than none."
)


@dataclass
class Agent:
    config: Config
    client: Client
    display: Display | None = None
    prompt_fn: PromptFn | None = None
    # Optional structured-event sink (programmatic API). Receives dicts:
    #   {"type": "turn", "turn": n}
    #   {"type": "tool", "name": str, "args": dict, "success": bool}
    #   {"type": "done", "success": bool, "turns": n}
    on_event: Callable[[dict[str, Any]], None] | None = None
    tool_ctx: ToolContext = field(init=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
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
            tool_profile=self.config.tool_profile,
            sandbox_image=self.config.sandbox_image,
            use_sandbox=self.config.use_sandbox,
            max_tool_output_chars=self._effective_output_cap(),
        )
        self.has_index = has_index(self.config.working_dir)
        project = detect_project(self.config.working_dir)
        system_prompt = build_system_prompt(
            self.config.working_dir,
            project,
            self.config.thinking,
            self.config.permission_mode,
            self.config.tool_profile,
        )
        self.messages.append({"role": "system", "content": system_prompt})

        if self.display is not None:
            self.display.stats.prompt_tokens += estimate_tokens(system_prompt)

        self._persist_new_messages()
        self._check_index_staleness()
        if self.display is not None:
            self.display.set_mode(self.config.permission_mode)

    def _is_constrained(self) -> bool:
        """True in the non-interactive modes, where a run is scored.

        `bench` and `yolo` run unattended against a turn budget; `edits` has a
        human present who can simply say "keep going".
        """
        return self.config.permission_mode in ("bench", "yolo")

    def _context_window(self) -> int:
        """Effective context window in tokens.

        Priority: explicit config override → value advertised by the endpoint →
        ``assumed_context_window``. The fallback matters: many local servers
        (LM Studio, llama.cpp) don't report ``context_length``, and without a
        value the compaction safety valve and dynamic history sizing were
        silently disabled — context then grew unbounded until the server errored.
        """
        if self.config.context_window > 0:
            return self.config.context_window
        reported = getattr(self.client, "context_window", 0) or 0
        if reported > 0:
            return reported
        return max(0, self.config.assumed_context_window)

    def _effective_output_cap(self) -> int:
        """Per-tool-result char cap, made window-aware for small models.

        The configured ``max_tool_output_chars`` (default 32k chars ≈ 9k tokens)
        can single-handedly blow an 8k–16k context window with one big
        ``read_file`` / ``run_command`` result. Cap a single result at ~1/8 of
        the effective window (chars ≈ tokens×3.5), floored so it stays useful
        and never raised above the configured value.
        """
        configured = self.config.max_tool_output_chars
        ctx_tokens = self._context_window()
        if ctx_tokens <= 0:
            return configured
        window_cap = int(ctx_tokens * 3.5 / 8)
        return max(4000, min(configured, window_cap))

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
        elif self.has_index and not has_graph(self.config.working_dir):
            # An index built before graphs existed leaves the graph tools
            # hidden and the user with no way to notice: `describe_staleness`
            # reports the index as fresh, because it is. Say it once.
            self.display.info(
                "[graph] no code graph yet — `explore`, `impact_of` and "
                "`repo_map` are hidden. Run /init to build one."
            )

    async def run(
        self, user_message: str, *, timeout: float | None = None,
    ) -> TaskResult:
        """Run one user turn to completion."""
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
        # Capture remaining messages before building result. The synthetic
        # live-context pair (plan status / notes) is a view-only artifact and
        # must never reach full_log (SFT export) or TaskResult.messages — it
        # would teach a `_squishy_context` tool call that does not exist.
        remaining = [
            m for m in self.messages[self._full_log_idx:]
            if not m.get(self._LIVE_CTX_MARKER)
        ]
        if remaining:
            self._full_log.extend(remaining)
            self._full_log_idx = len(self.messages)

        self._persist_new_messages()
        self._finish_session(st, status="completed" if success else "error", turns=turn)
        self._emit({"type": "done", "success": success, "turns": turn})
        return TaskResult(
            success=success, final_text=final_text, error=error,
            turns_used=turn,
            tokens_used=st.total_prompt_tokens + st.completion_tokens,
            files_created=sorted(st.files_created),
            files_edited=sorted(st.files_edited),
            commands_run=st.commands_run,
            elapsed_s=time.monotonic() - st.start,
            messages=[m for m in self.messages if not m.get(self._LIVE_CTX_MARKER)],
            empty_responses=st.empty_responses,
            prose_completions=st.prose_completions,
            tool_call_counts=dict(st.total_tool_calls),
            edit_failures=st.total_edit_failures,
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

    def _finish_session(
        self, st: LoopState, *, status: str = "completed", turns: int | None = None,
    ) -> None:
        if not self.session_id:
            return
        # turn_log is only populated in bench/yolo, so fall back to the actual
        # turn count passed by _build_result for interactive sessions (which
        # otherwise always recorded turns=0 in meta.json).
        turn_count = turns if turns is not None else (
            st.turn_log[-1].get("turn", 0) if st.turn_log else 0
        )
        try:
            from squishy.session import finish_session
            finish_session(
                self.session_id, status=status,
                turns=turn_count,
                tokens=st.total_prompt_tokens + st.completion_tokens,
                root=getattr(self.config, "session_dir", None),
            )
        except Exception:  # noqa: BLE001
            log.debug("session finish failed for %s", self.session_id, exc_info=True)

    # ------------------------------------------------------------------
    # Live-context injection (cache-stable system prefix)
    # ------------------------------------------------------------------
    #
    # Notes change between turns but we never want to touch
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

        Built from ``tool_ctx.notes`` (via ``render_notes``). Returns without
        appending when there are none — the prior pair has already been
        stripped, so the message list is back to canonical state.

        Must be called AFTER ``trim_history``/``compact_messages`` so the
        pair never participates in those routines.
        """
        self._strip_live_context_pair()
        if not self.tool_ctx.notes:
            return
        # A tail `system` message, not a fabricated (assistant tool_calls, tool
        # result) pair. The pair put a call to `_squishy_context` — a tool that
        # is in no schema we send — into the transcript the model reads, every
        # single turn. Models imitate their own history: on
        # msrest-for-python-43 the model duly called `_squishy_context` twice
        # and got `unknown tool` back both times. The harness demonstrated a
        # tool call and then refused it.
        #
        # `_build_result` already filtered this pair out of the SFT export for
        # exactly this reason ("it would teach a `_squishy_context` tool call
        # that does not exist") — but the live model was never covered by that
        # guard, only the training data downstream of it.
        #
        # A system message keeps every property the pair was chosen for: it is
        # appended at the tail so `messages[0]` is untouched and the vLLM prefix
        # cache survives, it carries no `tool_call_id` so it cannot orphan, and
        # `normalize_messages` passes non-assistant/tool roles through
        # unchanged. It is also not a user turn, so the "feedback belongs in the
        # tool result" rule is not at stake — this is standing context, not
        # feedback on an action.
        self.messages.append({
            "role": "system",
            "content": render_notes(self.tool_ctx.notes),
            self._LIVE_CTX_MARKER: True,
        })

    def _forget_invisible_reads(self) -> None:
        """Drop read-tracking for files whose content is no longer in history.

        ``read_file`` refuses a re-read once the model has already been served
        the same lines enough times. That is only fair while the model can
        still see them. ``trim_history`` keeps the last N messages, so a read
        from 30 turns ago is gone from the transcript while our bookkeeping
        still counts it — and the model gets refused for trying to recover
        content the harness deleted.

        ``agent_dispatch`` stamps ``_squishy_read_path`` on every successful
        read_file tool message, so the surviving set is exactly the reads the
        model can still see.
        """
        ctx = self.tool_ctx
        if not ctx.files_read_spans and not ctx.files_read_count:
            return

        visible: set[str] = set()
        for m in self.messages:
            rel = m.get("_squishy_read_path")
            if rel:
                visible.add(self._abs_read_path(str(rel)))

        for path in list(ctx.files_read_spans):
            if path not in visible:
                del ctx.files_read_spans[path]
        for path in list(ctx.files_read_count):
            if path not in visible:
                del ctx.files_read_count[path]
        # The cache and its hit counter are keyed by (abs_path, offset, limit).
        for key in list(ctx.files_read_meta):
            if key[0] not in visible:
                del ctx.files_read_meta[key]
        for key in list(ctx.read_cache_hits):
            if key[0] not in visible:
                del ctx.read_cache_hits[key]

    def _abs_read_path(self, rel: str) -> str:
        p = os.path.join(self.config.working_dir, rel)
        try:
            return os.path.realpath(p)
        except OSError:
            return p

    # ------------------------------------------------------------------
    # Prose completion handling
    # ------------------------------------------------------------------

    async def _handle_prose_completion(
        self, completion: CompletionResult, st: LoopState, turn: int, is_bench: bool,
    ) -> TaskResult | str:
        """Handle a completion with no tool calls.

        Returns a TaskResult to end the run, or "continue".

        A model that stops calling tools is done. The loop used to argue with
        it here — a finish gate that synthesized a pytest run, a single-shot
        "you saw a failure and didn't fix it" intercept, and the plan
        protocol's "call plan_task instead" — but the only case where the
        model genuinely has nothing to say is the empty response, so that is
        the only case still handled.
        """
        if not (completion.text or "").strip():
            st.consecutive_errors += 1
            st.empty_responses += 1
            if st.consecutive_errors >= self.config.max_consecutive_errors:
                msg = "model produced empty responses"
                if self.display:
                    self.display.error(msg)
                return self._build_result(st, success=False, error=msg, turn=turn)
            # Record the empty turn before nudging. It is a truthful record of
            # what the model produced, and without it the nudge is adjacent to
            # the previous user message — `_merge_adjacent_same_role` would
            # then glue the correction onto the task statement itself, which
            # is the anchor everything else is careful to protect.
            self.messages.append(prose_msg(""))
            inject_nudge(self, st, turn, (
                "[system] Your last response was empty. Either call a tool or "
                "reply with a plain-text summary of what you did."
            ), min_gap=0, force=True)
            return "continue"

        # A model that stops without having changed anything has not finished
        # the task; it has stopped. Under a turn budget that is a scored-zero
        # run, and the reference agent's whole margin came from refusing to
        # accept it: resume the same transcript, keep everything the model
        # learned, and let the turn budget be the only bound.
        #
        # Bounded by `max_turns` and nothing else. A count-based cap was tried
        # in the reference harness and is worse than it sounds -- each segment
        # ends after a handful of turns, so three nudges were spent by turn 21
        # of 50 and the run was declared over with 29 turns unused and no patch.
        #
        # This is an injected user turn, which the loop otherwise forbids. It
        # qualifies under the standing exception: there is no tool call to
        # attach it to, because not calling a tool is the thing being answered.
        # `bench` only, not `yolo`. Both are unattended, but only bench scores
        # a run by its diff -- in yolo a user can perfectly well ask a question
        # whose answer is prose, and nagging them to edit something would be
        # the harness inventing a goal the user did not set.
        if (
            is_bench
            and not st.files_edited
            # Creating a file is a fix too: an ImportError naming a symbol
            # from this repo is the task telling the model what to add.
            and not st.files_created
            and not st.shell_writes
            and turn < self.config.max_turns
        ):
            st.empty_patch_continues += 1
            self.messages.append(prose_msg(completion.text, completion.reasoning))
            if self.display:
                self.display.flush_streaming_text()
            self.messages.append({"role": "user", "content": EMPTY_PATCH_NUDGE})
            if self.display is not None:
                self.display.nudge(EMPTY_PATCH_NUDGE)
            return "continue"

        st.prose_completions += 1
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
        _is_constrained = self._is_constrained()

        if _is_constrained:
            for msg in self.messages:
                if msg.get("role") == "user" and not str(msg.get("content", "")).startswith("[system]"):
                    st.problem_files = extract_problem_files(str(msg.get("content", "")))
                    break

        # The schema is built once and never changes. Recomputing it per turn
        # only ever served gates that withdrew tools mid-run, and withdrawal
        # does not work: a model keeps calling a tool from its history after
        # it leaves the schema. A stable schema also keeps the prompt prefix
        # byte-identical turn over turn, so the server's prefix cache stays warm.
        _has_idx = has_index(self.config.working_dir)
        _has_graph = has_graph(self.config.working_dir)
        # `recall` is added back on top of a narrow profile when an index
        # exists. The graph tools are not: `graph` already lists `explore`,
        # `standard` shows them all, and quietly widening `minimal` would make
        # the two profiles incomparable in exactly the A/B this repo runs.
        schemas = openai_schemas(
            self.config.permission_mode,
            profile=self.config.tool_profile,
            extra_tools=frozenset({"recall"}) if _has_idx else frozenset(),
            has_index=_has_idx,
            has_graph=_has_graph,
        )

        for turn in range(1, self.config.max_turns + 1):
            self._active_turn = turn
            self._emit({"type": "turn", "turn": turn})

            # Publish the clock so tool results can carry edit pressure. This
            # replaced an injected `[system]` reminder every sixth turn: same
            # message, but paired with the call that earned it instead of
            # arriving out of band, and scaled to the budget actually left
            # rather than to a fixed period. See `tools/pressure.py`.
            self.tool_ctx.turns_used = turn
            self.tool_ctx.turn_budget = (
                self.config.max_turns if _is_constrained else 0
            )
            # Shell writes are edits the tools cannot see: under a shell-only
            # profile every change goes through `run_command`, and pressuring a
            # model that has already patched the file is how you get it undone.
            if st.files_edited or st.shell_writes:
                self.tool_ctx.source_edited = True

            self.tool_ctx.permission_mode = self.config.permission_mode

            # Strip any prior live-context pair so it never appears in
            # full_log snapshots, persistence, trim_history, or
            # compact_messages. It will be rebuilt below after trim/compact.
            self._strip_live_context_pair()

            # Snapshot new messages to full_log before trim/compaction can destroy them.
            new_msgs = self.messages[self._full_log_idx:]
            if new_msgs:
                self._full_log.extend(new_msgs)
                self._full_log_idx = len(self.messages)

            # Persist BEFORE trim reindexes the list and the blind index reset
            # below. Without this the just-appended user message is marked
            # persisted-but-never-written, and every path that `continue`s drops
            # its whole turn from the session log / --resume / training export.
            self._persist_new_messages()

            # Compaction + trim. Uses the *effective* window so endpoints that
            # don't advertise context_length still get compaction and dynamic
            # history sizing.
            ctx = self._context_window()
            did_compact = False
            if ctx > 0:
                compacted_msgs = await compact_messages(
                    self.messages, self.client,
                    context_limit=ctx,
                    threshold=self.config.compaction_threshold,
                )
                if len(compacted_msgs) < len(self.messages):
                    did_compact = True
                self.messages[:] = compacted_msgs
            # Dynamic history sizing: scale with the model's context window so
            # 128k models keep more history than 32k models. Bounded so we
            # don't blow up on absurdly long contexts.
            if ctx > 0:
                dyn_max = max(10, min(60, ctx // 4096))
                hist_cap = max(self.config.max_history_messages, dyn_max)
            else:
                hist_cap = self.config.max_history_messages
            self.messages[:] = trim_history(self.messages, max_messages=hist_cap)
            self._last_persisted_idx = len(self.messages)
            self._full_log_idx = len(self.messages)

            # Forget reads whose content trimming just dropped. The read guards
            # exist to stop a model circling over content it already has — so
            # they have to be measured against what the model can still SEE,
            # not against everything it ever read. Tracking that never expires
            # turns a correct re-read (the tool result scrolled out of history)
            # into "Refused: you have already read these lines", which was the
            # top tool failure in three consecutive sweeps. Compaction already
            # had this fix; trimming drops the same content without ever
            # setting did_compact, so it needs it too.
            self._forget_invisible_reads()

            # Rebuild the live-context pair AFTER trim/compact. The pair carries
            # notes for the model but is kept out of the canonical history so
            # the system prefix stays byte-stable turn-over-turn.
            self._refresh_live_context_pair()

            if did_compact:
                st.compaction_count += 1

                # Compaction summarizes file bodies away wholesale, so drop the
                # read tracking entirely rather than per-path.
                had_reads = bool(self.tool_ctx.files_read_count)
                self.tool_ctx.files_read_count.clear()
                self.tool_ctx.read_cache_hits.clear()
                self.tool_ctx.files_read_spans.clear()
                self.tool_ctx.files_read_meta.clear()

                # The one thing a tool result cannot tell the model: that the
                # harness rewrote its history underneath it.
                if had_reads or self.tool_ctx.notes:
                    inject_nudge(self, st, turn, (
                        "[system] Context was compacted — earlier file contents "
                        "are no longer in your history. Re-read any file you "
                        "need to edit rather than recalling it from memory."
                    ), min_gap=5, force=True)

            # Enforce the assistant/tool pairing invariant on the exact list we
            # send, so no malformed transcript reaches the endpoint.
            self.messages[:] = normalize_messages(self.messages)

            # LLM call. In interactive modes, show a spinner so a slow first
            # token / cold model doesn't look like a hung process.
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
                # are deterministic — no point retrying. Fail fast so the bench
                # harness moves on instead of burning the per-task budget.
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
            # seconds inside the call. Escalate when upstream is clearly
            # unstable so we don't burn the whole task_timeout in silence.
            call_retries = getattr(self.client, "last_call_retries", 0)
            if call_retries:
                st.cumulative_retries += call_retries
                if is_bench and st.cumulative_retries >= 24:
                    # Bail to the capture-on-error path so the workspace diff
                    # (any partial edits) still survives.
                    raise AgentTimeout(
                        f"retry storm: {st.cumulative_retries} cumulative "
                        f"upstream retries — aborting to preserve partial work"
                    )

            st.total_prompt_tokens += completion.prompt_tokens
            st.completion_tokens += completion.completion_tokens
            # Emitted per-turn so a harness can accumulate usage as it happens.
            # Sourcing metrics from TaskResult alone loses them on the common
            # weak-model paths (turn cap, timeout, upstream error).
            self._emit({
                "type": "usage", "turn": turn,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_prompt_tokens": st.total_prompt_tokens,
                "total_completion_tokens": st.completion_tokens,
                "tool_calls": len(completion.tool_calls or []),
            })

            # --- No tool calls: prose-only completion ---
            if not completion.tool_calls:
                result = await self._handle_prose_completion(completion, st, turn, is_bench)
                if isinstance(result, TaskResult):
                    return result
                continue

            # Tolerance layer: map alternate tool/parameter vocabularies (bash,
            # grep, str_replace, file_path, …) onto squishy's canonical tools so
            # a model fine-tuned on a different harness isn't penalized.
            for tc in completion.tool_calls:
                tc.name, tc.args = normalize_call(tc.name, tc.args)

            self.messages.append(
                assistant_msg(completion.text, completion.tool_calls, completion.reasoning)
            )

            # --- Loop detection ---
            # The only surviving loop check at this layer, and it ends the run
            # rather than trying to talk the model out of it. Per-tool echo
            # counters (read_file spans, run_command output hashes) handle the
            # softer cases inline, where the response reaches the model as a
            # tool result instead of an injected user turn.
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

            # --- Dispatch tools ---
            dispatched = 0
            for tc in completion.tool_calls:
                st.total_tool_calls[tc.name] = st.total_tool_calls.get(tc.name, 0) + 1

                outcome = await run_tool(self, turn, tc)
                dispatched += 1
                # Snapshot before the next dispatch overwrites it.
                pressure_tags = list(self.tool_ctx.last_pressure)
                if (
                    tc.name == "run_command"
                    and outcome.get("success")
                    and looks_like_file_write(str(tc.args.get("command", "")))
                ):
                    st.shell_writes += 1
                self._emit({
                    "type": "tool", "name": tc.name,
                    "args": dict(tc.args) if isinstance(tc.args, dict) else tc.args,
                    "success": bool(outcome.get("success")),
                    # Truncated: enough for a harness to bucket failures by
                    # cause without carrying whole tool outputs.
                    "error": str(outcome.get("error") or "")[:300],
                    # Which edit-pressure notices this result carried, so a
                    # harness can tell "warned and ignored" from "never fired".
                    "pressure": list(pressure_tags),
                })
                track_tool_outcome(self, st, tc, outcome, turn=turn)

                if st.consecutive_errors >= self.config.max_consecutive_errors:
                    msg = f"{self.config.max_consecutive_errors} consecutive tool failures — stopping."
                    if self.display:
                        self.display.error(msg)
                    return self._build_result(st, success=False, error=msg, turn=turn)

            # A successful edit is real progress — don't let identical
            # verification commands around it trip the loop detector.
            if _is_constrained and any(
                tc.name in ("edit_file", "write_file") for tc in completion.tool_calls
            ) and st.files_edited:
                st.consecutive_identical = 0
                st.last_call_key = ""

            # Per-turn event log (bench/yolo only).
            if _is_constrained:
                st.turn_log.append({
                    "turn": turn,
                    "tools": [
                        {"name": tc.name,
                         "args_summary": brief(tc) or str(tc.args.get("command", ""))[:80]}
                        for tc in completion.tool_calls
                    ],
                    "dispatched": dispatched,
                    "files_edited": len(st.files_edited),
                    "elapsed_s": round(time.monotonic() - st.start, 1),
                })

            self._persist_new_messages()

        # Max turns exhausted.
        msg = f"max turns ({self.config.max_turns}) reached"
        if self.display:
            self.display.warn(msg)
            self._sync_display_stats(st, self.config.max_turns)
        return self._build_result(st, success=False, error=msg, turn=self.config.max_turns)

    def _emit(self, event: dict[str, Any]) -> None:
        """Send a structured lifecycle event to the optional on_event sink.

        Never lets a caller's sink break the loop.
        """
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:  # noqa: BLE001
            log.debug("on_event sink raised", exc_info=True)

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
