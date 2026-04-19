"""Tests for trim_history tool_call / tool_result pairing."""

from __future__ import annotations

from squishy.context import trim_history


def _sys() -> dict:
    return {"role": "system", "content": "sys"}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_tc(call_id: str, name: str = "read_file", content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }


def _tool_result(call_id: str, name: str = "read_file", content: str = "{}") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def test_under_limit_preserves_valid_pairs() -> None:
    msgs = [
        _sys(),
        _user("hi"),
        _assistant_tc("c1"),
        _tool_result("c1"),
        {"role": "assistant", "content": "done"},
    ]
    out = trim_history(msgs, max_messages=20)
    assert out == msgs


def test_orphan_tool_result_dropped() -> None:
    # Tool result with no matching assistant tool_call → dropped.
    msgs = [
        _sys(),
        _user("hi"),
        _tool_result("ghost"),
        {"role": "assistant", "content": "done"},
    ]
    out = trim_history(msgs, max_messages=20)
    assert not any(m.get("role") == "tool" for m in out)


def test_orphan_assistant_tool_call_cleaned() -> None:
    # Assistant has tool_calls but no matching tool result.
    msgs = [
        _sys(),
        _user("hi"),
        _assistant_tc("c1", content="partial"),
        # No tool result for c1
        {"role": "assistant", "content": "done"},
    ]
    out = trim_history(msgs, max_messages=20)
    # The partial assistant message keeps its content but tool_calls is stripped
    # (or the message is dropped entirely if content was empty).
    assert not any("tool_calls" in m and m.get("tool_calls") for m in out)


def test_tail_truncation_drops_orphaned_pair_members() -> None:
    # Build a conversation that exceeds the budget and forces the trim to
    # keep only the last N. The earliest tool-call pair should end up either
    # kept intact or dropped — never half.
    msgs = [_sys(), _user("original")]
    for i in range(20):
        msgs.append(_assistant_tc(f"c{i}"))
        msgs.append(_tool_result(f"c{i}"))
    msgs.append({"role": "assistant", "content": "final"})

    out = trim_history(msgs, max_messages=10)

    # Every tool result has a matching assistant tool_call in the output.
    assistant_ids = set()
    for m in out:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                assistant_ids.add(tc["id"])
    for m in out:
        if m.get("role") == "tool":
            assert m["tool_call_id"] in assistant_ids

    # Every assistant tool_call has a matching tool result.
    tool_ids = {m["tool_call_id"] for m in out if m.get("role") == "tool"}
    for m in out:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                assert tc["id"] in tool_ids


def test_preserves_system_and_first_user() -> None:
    msgs = [_sys()] + [{"role": "user", "content": f"u{i}"} for i in range(30)]
    out = trim_history(msgs, max_messages=10)
    assert out[0] == _sys()
    # First non-system user message is preserved at index 1
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "u0"
