"""Tests for the background ModeCycler.

We can't easily exercise the real TTY path under pytest, so we drive the
parser via ModeCycler.feed and verify the on_cycle callback fires once
per recognized escape sequence.
"""

from __future__ import annotations

import pytest

from squishy.async_input import ModeCycler


def test_shift_tab_cycles_once():
    calls = []
    cycler = ModeCycler(on_cycle=lambda: calls.append(1))
    fired = cycler.feed(b"\x1b[Z")
    assert fired == 1
    assert calls == [1]


def test_repeated_shift_tabs_cycle_each_time():
    calls = []
    cycler = ModeCycler(on_cycle=lambda: calls.append(1))
    cycler.feed(b"\x1b[Z\x1b[Z\x1b[Z")
    assert calls == [1, 1, 1]


def test_alt_shift_tab_sequences_recognized():
    calls = []
    cycler = ModeCycler(on_cycle=lambda: calls.append(1))
    cycler.feed(b"\x1b\x09")
    cycler.feed(b"\x1bOZ")
    assert calls == [1, 1]


def test_unrelated_keys_do_not_cycle():
    calls = []
    cycler = ModeCycler(on_cycle=lambda: calls.append(1))
    cycler.feed(b"abc\x1b[Aescape-up")
    assert calls == []


def test_cycle_callback_exception_is_swallowed():
    """If the cycle callback raises, the listener must keep working."""
    calls = []

    def cb():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    cycler = ModeCycler(on_cycle=cb)
    cycler.feed(b"\x1b[Z\x1b[Z")
    # Both keystrokes were processed; the parser didn't get stuck.
    assert calls == [1, 1]


def test_buffer_is_bounded():
    """A flood of unrelated bytes must not let the buffer grow unbounded."""
    cycler = ModeCycler(on_cycle=lambda: None)
    cycler.feed(b"x" * 1024)
    assert len(cycler._buf) <= 64


def test_paused_is_safe_when_inactive():
    """paused() must not crash when the cycler was never started."""
    cycler = ModeCycler(on_cycle=lambda: None)
    with cycler.paused():
        pass


@pytest.mark.asyncio
async def test_async_context_manager_no_tty():
    """In a non-TTY pytest environment, the cycler is a no-op but the
    async-context interface still works without raising."""
    cycler = ModeCycler(on_cycle=lambda: None)
    async with cycler:
        pass
