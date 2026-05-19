"""Tests for the v6e/v6f UX-hardening pass.

Covers behaviors added so the user can see what's happening and steer
the agent mid-task:

  - ``Display.nudge`` surfaces system nudges (otherwise invisible
    ``[system] ...`` instructions injected as user-role messages)
  - ``Display.reset_streaming`` drops in-flight buffer without printing
    (used by the client retry path so partial first-attempt text isn't
    concatenated with the retry's full response)
  - ``Display.start_thinking`` / ``stop_thinking`` spinner idempotency
  - ``Display.mode_changed`` prints escalation warnings when leaving
    plan mode
  - ``file_browser.inject_references_with_missing`` reports @typo paths
    so the CLI can warn the user
  - ``Client.complete(on_retry=...)`` fires the callback before each
    backoff sleep with (attempt_number, max_attempts, exception)
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import httpx
from rich.console import Console
from tenacity import wait_none

import squishy.client as client_mod
from squishy.client import Client
from squishy.display import Display
from squishy.file_browser import (
    inject_references_with_missing,
    parse_references_with_missing,
)


def _capture_display() -> tuple[Display, io.StringIO]:
    buf = io.StringIO()
    d = Display()
    d.console = Console(file=buf, force_terminal=False, width=120)
    return d, buf


# -- Display.nudge ------------------------------------------------------------


def test_nudge_renders_in_panel():
    d, buf = _capture_display()
    d.nudge("[system] please run the failing test before claiming success")
    out = buf.getvalue()
    assert "system nudge" in out
    # Leading "[system] " marker is stripped before rendering.
    assert "please run the failing test" in out
    assert "[system]" not in out


def test_nudge_caps_long_content():
    d, buf = _capture_display()
    long_msg = "x" * 1000
    d.nudge(long_msg)
    out = buf.getvalue()
    # 600-char cap + " …" suffix; original 1000-char wall must not survive.
    assert "x" * 1000 not in out
    assert "…" in out


def test_nudge_empty_content_is_noop():
    d, buf = _capture_display()
    d.nudge("")
    assert buf.getvalue() == ""


def test_nudge_escapes_rich_markup():
    """Nudges carry arbitrary content (file paths, error messages) that
    may contain ``[`` characters Rich would otherwise treat as broken
    markup."""
    d, buf = _capture_display()
    d.nudge("edit failed at path foo[bar].py — list[int] mismatch")
    out = buf.getvalue()
    assert "foo[bar].py" in out
    assert "list[int]" in out


# -- Display.reset_streaming --------------------------------------------------


def test_reset_streaming_drops_buffer_without_printing():
    """Used by the client retry path so the first attempt's partial
    stream doesn't get concatenated with the retry's full response."""
    d, buf = _capture_display()
    d.streaming_text_chunk("partial first attempt ")
    assert d._stream_buffer == "partial first attempt "
    d.reset_streaming()
    assert d._stream_buffer == ""
    assert d._live is None
    assert d._use_live is False
    # Crucially: a second stream after reset starts fresh — no
    # concatenation with the dropped partial.
    d.streaming_text_chunk("clean retry text")
    assert d._stream_buffer == "clean retry text"


def test_reset_streaming_is_safe_when_idle():
    d, buf = _capture_display()
    # No active stream — must not raise.
    d.reset_streaming()
    assert d._stream_buffer == ""


# -- Display.start_thinking / stop_thinking ----------------------------------


def test_start_thinking_is_idempotent():
    d, _ = _capture_display()
    d.start_thinking("waiting")
    first = d._spinner
    d.start_thinking("waiting")  # second call should be a no-op
    assert d._spinner is first
    d.stop_thinking()
    assert d._spinner is None


def test_stop_thinking_is_safe_when_idle():
    d, _ = _capture_display()
    # Never started — must not raise.
    d.stop_thinking()
    assert d._spinner is None


def test_start_thinking_skipped_during_active_stream():
    """If a stream is in flight, the spinner must not start — they share
    the Live channel and would fight for the same screen rows."""
    d, _ = _capture_display()
    d.streaming_text_chunk("hello")
    assert d._live is not None
    d.start_thinking("thinking")
    assert d._spinner is None  # refused because Live is active
    d.flush_streaming_text()


# -- Display.mode_changed -----------------------------------------------------


def test_mode_changed_warns_on_plan_to_edits_escalation():
    d, buf = _capture_display()
    d.set_mode("plan")
    d.mode_changed("edits")
    out = buf.getvalue()
    assert "mode → edits" in out
    # The whole point of the warning: write tools are now allowed.
    assert "write tools" in out


def test_mode_changed_warns_on_yolo():
    d, buf = _capture_display()
    d.set_mode("edits")
    d.mode_changed("yolo")
    out = buf.getvalue()
    assert "mode → yolo" in out
    assert "yolo" in out  # yolo-specific warning line


def test_mode_changed_no_warning_for_edits_to_plan_descalation():
    d, buf = _capture_display()
    d.set_mode("edits")
    d.mode_changed("plan")
    out = buf.getvalue()
    assert "mode → plan" in out
    # De-escalation: no warning needed because plan removes write tools.
    assert "write tools" not in out


# -- file_browser missing-reference tracking ---------------------------------


def test_parse_references_with_missing_separates_found_and_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        good = os.path.join(tmpdir, "exists.py")
        with open(good, "w") as f:
            f.write("ok")
        text = "Check @exists.py and @typo.py and @also_missing.py"
        found, missing = parse_references_with_missing(text, tmpdir)
        assert len(found) == 1
        assert found[0].path == "exists.py"
        assert missing == ["typo.py", "also_missing.py"]


def test_inject_references_with_missing_strips_literal_at_paths():
    """A literal ``@typo.py`` left in the prompt confuses weak models,
    so the inject path replaces it with an inline marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        text = "Fix the bug in @nonexistent.py please"
        result, refs, missing = inject_references_with_missing(text, tmpdir)
        assert refs == []
        assert missing == ["nonexistent.py"]
        assert "@nonexistent.py" not in result
        assert "[file not found: nonexistent.py]" in result


def test_inject_references_with_missing_preserves_found_refs():
    with tempfile.TemporaryDirectory() as tmpdir:
        good = os.path.join(tmpdir, "app.py")
        with open(good, "w") as f:
            f.write("def f(): pass")
        text = "Read @app.py and @missing.py"
        result, refs, missing = inject_references_with_missing(text, tmpdir)
        assert len(refs) == 1
        assert missing == ["missing.py"]
        assert "@app.py" not in result
        assert "@missing.py" not in result
        assert "def f(): pass" in result
        assert "[file not found: missing.py]" in result


# -- Client.on_retry callback -------------------------------------------------


class _FlakyCompletions:
    def __init__(self, fail_n: int, exc: Exception) -> None:
        self._remaining = fail_n
        self._exc = exc

    async def create(self, **_: Any):
        if self._remaining > 0:
            self._remaining -= 1
            raise self._exc
        return _fake_response()


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


def _fake_response():
    class _Msg:
        content = "ok"
        tool_calls = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class _R:
        choices = [_Choice()]
        usage = _Usage()

    return _R()


async def test_client_on_retry_fires_per_backoff():
    """on_retry must be invoked once per backoff sleep with
    (attempt_number, max_attempts, exception)."""
    client = Client(
        base_url="http://example.invalid/v1",
        api_key="local",
        model="fake",
        max_retries=4,
    )
    flaky = _FlakyCompletions(fail_n=2, exc=httpx.ConnectError("boom"))
    client._client.chat = _Chat(flaky)  # type: ignore[attr-defined]

    calls: list[tuple[int, int, BaseException]] = []

    def _on_retry(attempt: int, max_attempts: int, exc: BaseException) -> None:
        calls.append((attempt, max_attempts, exc))

    original_wait = client_mod.wait_exponential
    client_mod.wait_exponential = lambda **_: wait_none()  # type: ignore[assignment]
    try:
        result = await client.complete(
            [{"role": "user", "content": "hi"}], [],
            stream=False, on_retry=_on_retry,
        )
    finally:
        client_mod.wait_exponential = original_wait
        await client.aclose()

    assert result.text == "ok"
    # Two failures → two before_sleep fires → on_retry called twice.
    assert len(calls) == 2
    assert calls[0][0] == 1  # attempt_number of the failed try
    assert calls[0][1] == 4  # max_attempts surfaced
    assert isinstance(calls[0][2], httpx.ConnectError)
    # last_call_retries should match what the callback observed.
    assert client.last_call_retries == 2


async def test_client_on_retry_not_called_on_success():
    client = Client(
        base_url="http://example.invalid/v1",
        api_key="local",
        model="fake",
        max_retries=4,
    )
    flaky = _FlakyCompletions(fail_n=0, exc=httpx.ConnectError("never"))
    client._client.chat = _Chat(flaky)  # type: ignore[attr-defined]

    calls: list[int] = []

    def _on_retry(attempt: int, max_attempts: int, exc: BaseException) -> None:
        calls.append(attempt)

    try:
        await client.complete(
            [{"role": "user", "content": "hi"}], [],
            stream=False, on_retry=_on_retry,
        )
    finally:
        await client.aclose()

    assert calls == []
    assert client.last_call_retries == 0


async def test_client_on_retry_callback_errors_dont_break_retry():
    """A flaky display callback must not abort the retry loop."""
    client = Client(
        base_url="http://example.invalid/v1",
        api_key="local",
        model="fake",
        max_retries=4,
    )
    flaky = _FlakyCompletions(fail_n=2, exc=httpx.ConnectError("boom"))
    client._client.chat = _Chat(flaky)  # type: ignore[attr-defined]

    def _broken_callback(attempt: int, max_attempts: int, exc: BaseException) -> None:
        raise RuntimeError("display blew up")

    original_wait = client_mod.wait_exponential
    client_mod.wait_exponential = lambda **_: wait_none()  # type: ignore[assignment]
    try:
        result = await client.complete(
            [{"role": "user", "content": "hi"}], [],
            stream=False, on_retry=_broken_callback,
        )
    finally:
        client_mod.wait_exponential = original_wait
        await client.aclose()

    # Retry budget still ran to completion despite the broken callback.
    assert result.text == "ok"
    assert client.last_call_retries == 2
