"""Nudge primitives for the agent loop.

This module used to hold a quality gate, a goal-drift detector, a
test-failure summarizer and several edit-pressure heuristics. They are gone
on purpose. Every one of them spoke to the model by appending an
out-of-band ``[system]`` user message, and the quality gate went further:
it discarded the assistant's tool calls and let ``normalize_messages``
erase the turn, so the model was corrected for an action its own history no
longer contained. Measured on SWE-rebench, harness-generated refusals were
the single largest failure bucket.

The rule that replaced them: **feedback belongs in the tool result.** A tool
result is causally paired with the call that produced it, is what the model
is trained to read, and cannot desynchronize the transcript. Loop-breaking
now lives in the tools themselves — ``read_file``'s span cache and
``run_command``'s echo counter both answer inline.

What survives here is the injection primitive itself, for the handful of
things a tool result genuinely cannot say: the model returned nothing at
all, or the harness changed the transcript under it (compaction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from squishy.agent_state import LoopState

if TYPE_CHECKING:
    from squishy.agent import Agent


def can_nudge(st: LoopState, turn: int, min_gap: int = 2, max_total: int = 12) -> bool:
    """Return True if enough turns have passed since the last nudge and
    the total nudge count hasn't exceeded the cap."""
    if st.total_nudges >= max_total:
        return False
    return turn - st.last_nudge_turn >= min_gap


def record_nudge(st: LoopState, turn: int) -> None:
    """Mark that a nudge was injected this turn."""
    if turn == st.last_nudge_turn:
        st.nudges_this_turn += 1
    else:
        st.nudges_this_turn = 1
    st.last_nudge_turn = turn
    st.total_nudges += 1


# Even control-flow nudges (force=True, min_gap=0) are capped per turn.
# Non-forced nudges are already limited to one per turn by ``min_gap``; without
# this, several forced gates could each append a [system] message in the same
# turn, spending context and (on strict-alternation templates) stacking
# consecutive user turns.
MAX_NUDGES_PER_TURN = 2


def inject_nudge(
    agent: Agent, st: LoopState, turn: int, content: str,
    *, min_gap: int = 2, force: bool = False,
) -> bool:
    """Inject a system nudge if under the cap. Returns True if injected.

    Mirrors the nudge to ``agent.display.nudge`` so the user can see every
    correction the harness sends to the model — otherwise agent behavior
    changes silently in response to invisible system messages.

    Callers must only use this between complete turns. Injecting between an
    assistant ``tool_calls`` message and its paired tool results leaves an
    orphan that ``normalize_messages`` repairs by deleting the assistant
    turn.
    """
    # Hard ceiling: even forced nudges stop after 2x the normal cap.
    hard_cap = agent.config.max_system_nudges * 2
    if st.total_nudges >= hard_cap:
        return False
    # Per-turn ceiling applies to forced nudges too (see MAX_NUDGES_PER_TURN).
    if turn == st.last_nudge_turn and st.nudges_this_turn >= MAX_NUDGES_PER_TURN:
        return False
    if not force and not can_nudge(st, turn, min_gap=min_gap):
        return False
    agent.messages.append({"role": "user", "content": content})
    record_nudge(st, turn)
    if agent.display is not None:
        agent.display.nudge(content)
    return True
