"""Byte-stability tests for the system prompt prefix.

These tests pin the contract that ``Agent._refresh_live_context_pair`` must
NOT mutate ``messages[0]`` — the system message must stay byte-stable
turn-over-turn so vLLM's prefix cache stays warm during local serving.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

from squishy.agent import Agent
from squishy.plan_state import PlanState


def _make_pair_refresher(
    messages: list[dict[str, Any]],
    plan: PlanState | None = None,
    notes: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Build a thin object that exposes the same surface
    ``_refresh_live_context_pair`` reads from on a real Agent."""
    obj = SimpleNamespace(
        messages=messages,
        tool_ctx=SimpleNamespace(plan=plan, notes=notes or {}),
        # Mirror the class-level constants so the unbound methods can read
        # them via ``self.<name>``.
        _LIVE_CTX_MARKER=Agent._LIVE_CTX_MARKER,
        _LIVE_CTX_TOOL_NAME=Agent._LIVE_CTX_TOOL_NAME,
        _LIVE_CTX_CALL_ID=Agent._LIVE_CTX_CALL_ID,
    )
    # Bind the unbound methods so they treat ``obj`` as ``self``.
    obj._strip_live_context_pair = (
        lambda: Agent._strip_live_context_pair(obj)  # type: ignore[arg-type]
    )
    obj._refresh_live_context_pair = (
        lambda: Agent._refresh_live_context_pair(obj)  # type: ignore[arg-type]
    )
    return obj


def _sys_hash(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha1(messages[0]["content"].encode()).hexdigest()


def test_system_prefix_is_byte_stable_across_plan_progress():
    """The cache-warmth invariant: plan progressing must NOT mutate
    ``messages[0]``. Hash must be identical before and after a refresh
    even when plan status changes between calls."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "fixed system prompt content"},
        {"role": "user", "content": "first user message"},
    ]
    plan = PlanState.create(
        problem="bug", solution="fix",
        steps=["read file", "edit file", "run tests"],
    )
    plan.approved = True
    refresher = _make_pair_refresher(messages, plan=plan)

    before_hash = _sys_hash(messages)
    refresher._refresh_live_context_pair()
    after_first = _sys_hash(messages)

    # Mark a step done — plan-status rendering changes.
    plan.steps[0].apply(status="done")
    refresher._refresh_live_context_pair()
    after_second = _sys_hash(messages)

    assert before_hash == after_first == after_second, (
        "system message content must be byte-stable across refresh calls"
    )


def test_refresh_appends_well_formed_pair_at_tail():
    """The injected pair must be assistant+tool with matching
    ``tool_call_id``, tagged with the live-ctx marker."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    plan = PlanState.create(problem="p", solution="s", steps=["s1"])
    refresher = _make_pair_refresher(messages, plan=plan)

    refresher._refresh_live_context_pair()

    assert len(messages) == 4
    assistant, tool = messages[-2], messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant.get(Agent._LIVE_CTX_MARKER) is True
    tcs = assistant.get("tool_calls")
    assert isinstance(tcs, list) and len(tcs) == 1
    assert tcs[0]["id"] == Agent._LIVE_CTX_CALL_ID
    assert tcs[0]["function"]["name"] == Agent._LIVE_CTX_TOOL_NAME

    assert tool["role"] == "tool"
    assert tool.get(Agent._LIVE_CTX_MARKER) is True
    assert tool["tool_call_id"] == Agent._LIVE_CTX_CALL_ID
    assert tool["name"] == Agent._LIVE_CTX_TOOL_NAME


def test_refresh_strips_prior_pair_before_appending():
    """Two consecutive refreshes must not stack — the pair count stays at one."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    plan = PlanState.create(problem="p", solution="s", steps=["s1"])
    refresher = _make_pair_refresher(messages, plan=plan)

    refresher._refresh_live_context_pair()
    refresher._refresh_live_context_pair()
    refresher._refresh_live_context_pair()

    marked = [m for m in messages if m.get(Agent._LIVE_CTX_MARKER)]
    assert len(marked) == 2, "exactly one (assistant, tool) pair must be present"


def test_refresh_noop_when_plan_and_notes_empty():
    """No pair should be appended when there is nothing to surface."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    refresher = _make_pair_refresher(messages, plan=None, notes={})

    refresher._refresh_live_context_pair()

    assert len(messages) == 2
    assert not any(m.get(Agent._LIVE_CTX_MARKER) for m in messages)


def test_refresh_includes_notes_when_present():
    """Notes alone (no plan) still produce a pair carrying the notes block."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]
    notes = {"finding-1": "the bug is in foo.py line 42"}
    refresher = _make_pair_refresher(messages, plan=None, notes=notes)

    refresher._refresh_live_context_pair()

    tool = messages[-1]
    assert tool["role"] == "tool"
    assert "foo.py line 42" in tool["content"]


def test_strip_removes_pair_without_affecting_other_messages():
    """``_strip_live_context_pair`` must only touch tagged messages."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
    ]
    plan = PlanState.create(problem="p", solution="s", steps=["s1"])
    refresher = _make_pair_refresher(messages, plan=plan)
    refresher._refresh_live_context_pair()
    assert len(messages) == 5

    refresher._strip_live_context_pair()
    assert len(messages) == 3
    assert messages[0]["content"] == "sys"
    assert messages[1]["content"] == "u"
    assert messages[2]["content"] == "a1"
