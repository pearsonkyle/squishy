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

def test_tool_schemas_differ_by_mode():
    """Tool schemas should differ between plan and edits modes."""
    from squishy.tools import openai_schemas

    plan_schemas = openai_schemas("plan")
    edits_schemas = openai_schemas("edits")
    yolo_schemas = openai_schemas("yolo")
    bench_schemas = openai_schemas("bench")

    plan_names = {s["function"]["name"] for s in plan_schemas}
    edits_names = {s["function"]["name"] for s in edits_schemas}
    yolo_names = {s["function"]["name"] for s in yolo_schemas}
    bench_names = {s["function"]["name"] for s in bench_schemas}

    # Plan mode should NOT have edit_file or write_file.
    assert "edit_file" not in plan_names
    assert "write_file" not in plan_names
    # Edits and yolo should have edit_file.
    assert "edit_file" in edits_names
    assert "edit_file" in yolo_names
    assert "edit_file" in bench_names
    # All modes should have read_file.
    assert "read_file" in plan_names
    assert "read_file" in edits_names


async def test_mode_switch_between_runs(tmp_path):
    """Running with different permission modes should use different tool schemas."""
    schemas_seen: list[list[dict]] = []

    class _SchemaCapture(_ScriptedClient):
        async def complete(self, messages, tools, **kwargs):
            schemas_seen.append(tools)
            return CompletionResult(text="done.", tool_calls=[])

    # Each Squishy creates its own Client in __post_init__, so we use
    # side_effect to return a fresh capture instance each time.
    with patch("squishy.api.Client", side_effect=lambda **kw: _SchemaCapture([])):
        # First run in plan mode
        sq1 = Squishy(model="fake", permission_mode="plan")
        await sq1.run("explore code", working_dir=str(tmp_path))
        await sq1.aclose()

        # Second run in edits mode
        sq2 = Squishy(model="fake", permission_mode="edits")
        await sq2.run("fix bug", working_dir=str(tmp_path))
        await sq2.aclose()

    assert len(schemas_seen) >= 2
    # The plan run may produce multiple completions, so check the first
    # and last captured schemas instead of indices 0 and 1.
    plan_tools = {s["function"]["name"] for s in schemas_seen[0]}
    edits_tools = {s["function"]["name"] for s in schemas_seen[-1]}
    # Plan shouldn't have write tools, edits should.
    assert "edit_file" not in plan_tools
    assert "edit_file" in edits_tools
