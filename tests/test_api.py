"""Tests for the programmatic Squishy facade."""
 
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from squishy.api import Squishy
from squishy.client import CompletionResult, ToolCall


class _ScriptedClient:
    """Drop-in replacement for Client; ignores real network."""
 
    def __init__(self, script: list[CompletionResult]) -> None:
        self._script = script
        self._i = 0
        self.closed = False
 
    async def health(self) -> bool:
        return True
 
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: Any = None,
        on_retry: Any = None,
    ) -> CompletionResult:
        if self._i >= len(self._script):
            return CompletionResult(text="done.", tool_calls=[])
        result = self._script[self._i]
        self._i += 1
        # If on_text is provided, stream the text character-by-character so
        # callback wiring is exercised.
        if on_text and result.text:
            for ch in result.text:
                maybe = on_text(ch)
                if maybe is not None:
                    await maybe
        return result
 
    async def aclose(self) -> None:
        self.closed = True
 
 
async def test_squishy_run_roundtrip(tmp_path):
    script = [
        CompletionResult(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="write_file",
                    args={"path": "greet.py", "content": "print('hi')\n"},
                )
            ]
        ),
        CompletionResult(text="wrote greet.py", tool_calls=[]),
    ]
    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake") as sq:
            result = await sq.run("create greet.py", working_dir=str(tmp_path))
 
    assert result.success
    assert "greet.py" in result.files_created
    assert (tmp_path / "greet.py").read_text() == "print('hi')\n"
 
 
async def test_squishy_on_text_callback(tmp_path):
    script = [CompletionResult(text="hello world", tool_calls=[])]
    chunks: list[str] = []
 
    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake") as sq:
            result = await sq.run(
                "say hi", working_dir=str(tmp_path), on_text=chunks.append
            )
 
    assert result.success
    assert "".join(chunks) == "hello world"
 
 
async def test_squishy_rejects_invalid_permission_mode():
    with pytest.raises(ValueError):
        Squishy(model="x", permission_mode="bogus")
 
 
async def test_squishy_aclose_closes_client(tmp_path):
    client = _ScriptedClient([CompletionResult(text="k", tool_calls=[])])
    with patch("squishy.api.Client", return_value=client):
        sq = Squishy(model="fake")
        await sq.aclose()
 
    assert client.closed


# -- Mode switching tests ------------------------------------------------------



async def test_chat_session_multi_turn(tmp_path):
    """ChatSession should persist agent state across multiple send() calls."""
    script = [
        # Turn 1: create a file
        CompletionResult(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="write_file",
                    args={"path": "app.py", "content": "x = 1\n"},
                )
            ]
        ),
        CompletionResult(text="Created app.py", tool_calls=[]),
        # Turn 2: edit the file (agent remembers it exists)
        CompletionResult(
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="edit_file",
                    args={"path": "app.py", "old_str": "x = 1", "new_str": "x = 42"},
                )
            ]
        ),
        CompletionResult(text="Updated x to 42", tool_calls=[]),
    ]
    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake") as sq:
            async with sq.chat(working_dir=str(tmp_path)) as session:
                r1 = await session.send("create app.py")
                assert r1.success
                assert "app.py" in r1.files_created

                r2 = await session.send("change x to 42")
                assert r2.success

    assert (tmp_path / "app.py").read_text() == "x = 42\n"


async def test_chat_session_on_text_callback(tmp_path):
    """ChatSession should forward text chunks to the on_text callback."""
    script = [CompletionResult(text="hello from chat", tool_calls=[])]
    chunks: list[str] = []

    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake") as sq:
            async with sq.chat(
                working_dir=str(tmp_path), on_text=chunks.append
            ) as session:
                await session.send("say hi")

    assert "".join(chunks) == "hello from chat"




def test_squishy_api_fields_cover_config():
    """Regression: every Config field that can be set programmatically
    should also be settable on Squishy.

    This protects against latent gaps where Config grows a knob but the
    public API silently drops it (the v25 audit found 4 such fields).
    """
    from dataclasses import fields as dc_fields

    from squishy.api import Squishy
    from squishy.config import Config

    config_field_names = {
        f.name for f in dc_fields(Config)
        # working_dir is set per-call via Squishy.run(working_dir=...)
        if f.name != "working_dir"
    }
    squishy_field_names = {
        f.name for f in dc_fields(Squishy)
        if not f.name.startswith("_")
    }
    missing = config_field_names - squishy_field_names
    assert not missing, (
        f"Squishy dataclass missing Config fields: {sorted(missing)}"
    )


def test_bench_tools_excludes_fetch_url():
    """B6: bench mode is air-gapped so fetch_url should not appear in
    the tool schema."""
    from squishy.tool_restrictions import BENCH_TOOLS
    assert "fetch_url" not in BENCH_TOOLS


# -- Phase 5: API ergonomics --------------------------------------------------

async def test_env_var_precedence_when_not_overridden(monkeypatch):
    """A caller who sets env vars (and doesn't pass explicit values) gets them
    honored instead of silently overridden by facade defaults."""
    monkeypatch.setenv("SQUISHY_BASE_URL", "http://envhost:9999/v1")
    monkeypatch.setenv("SQUISHY_API_KEY", "env-secret")
    monkeypatch.setenv("SQUISHY_MODEL", "env-model")
    with patch("squishy.api.Client", return_value=_ScriptedClient([])):
        sq = Squishy()  # no explicit connection args
        cfg = sq._make_config(None)
    assert cfg.base_url == "http://envhost:9999/v1"
    assert cfg.api_key == "env-secret"
    assert cfg.model == "env-model"


async def test_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("SQUISHY_BASE_URL", "http://envhost:9999/v1")
    with patch("squishy.api.Client", return_value=_ScriptedClient([])):
        sq = Squishy(model="m", base_url="http://explicit:1/v1")
        cfg = sq._make_config(None)
    assert cfg.base_url == "http://explicit:1/v1"




async def test_per_run_mode_override_rejects_bad_mode(tmp_path):
    with patch("squishy.api.Client", return_value=_ScriptedClient([])):
        sq = Squishy(model="fake")
        with pytest.raises(ValueError):
            sq._make_config(str(tmp_path), "bogus")  # type: ignore[arg-type]


async def test_on_event_emits_turn_tool_done(tmp_path):
    script = [
        CompletionResult(tool_calls=[ToolCall(id="c1", name="write_file",
                         args={"path": "a.py", "content": "x"})]),
        CompletionResult(text="done", tool_calls=[]),
    ]
    events: list[dict] = []
    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake", permission_mode="yolo") as sq:
            await sq.run("write a.py", working_dir=str(tmp_path), on_event=events.append)
    types = [e["type"] for e in events]
    assert "turn" in types
    assert "done" in types
    tool_events = [e for e in events if e["type"] == "tool"]
    assert any(e["name"] == "write_file" and e["success"] for e in tool_events)


async def test_on_text_exception_is_logged_not_fatal(tmp_path):
    """A throwing on_text must not crash the run (and is no longer swallowed
    silently — it's logged)."""
    def boom(_chunk):
        raise RuntimeError("callback boom")
    script = [CompletionResult(text="hello", tool_calls=[])]
    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake") as sq:
            result = await sq.run("hi", working_dir=str(tmp_path), on_text=boom)
    assert result.success


async def test_async_on_text_is_scheduled_not_dropped(tmp_path):
    seen: list[str] = []
    async def async_cb(chunk):
        seen.append(chunk)
    script = [CompletionResult(text="abc", tool_calls=[])]
    with patch("squishy.api.Client", return_value=_ScriptedClient(script)):
        async with Squishy(model="fake") as sq:
            await sq.run("hi", working_dir=str(tmp_path), on_text=async_cb)
    # Let scheduled callback tasks run.
    import asyncio
    await asyncio.sleep(0)
    assert "".join(seen) == "abc"
