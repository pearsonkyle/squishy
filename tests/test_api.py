"""Tests for the programmatic Squishy facade."""
 
from __future__ import annotations
 
from typing import Any
from unittest.mock import patch
 
import pytest
 
from squishy.api import Squishy
from squishy.client import CompletionResult, ToolCall
 
pytestmark = pytest.mark.asyncio
 
 
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
