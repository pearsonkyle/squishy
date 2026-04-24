"""Retry and error-translation tests for the async client."""
 
from __future__ import annotations
 
from typing import Any
 
import httpx
import pytest
from tenacity import wait_none
 
import squishy.client as client_mod
from squishy.client import Client
from squishy.errors import LLMError
 
pytestmark = pytest.mark.asyncio
 
 
class _FlakyCompletions:
    """Fake ``chat.completions.create`` that fails the first N calls then succeeds."""
 
    def __init__(self, fail_n: int, exc: Exception) -> None:
        self._remaining = fail_n
        self._exc = exc
        self.calls = 0
 
    async def create(self, **_: Any):
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._exc
        return _fake_response()
 
 
class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions
 
 
class _Models:
    async def list(self) -> Any:
        return object()
 
 
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
 
 
def _client_with_fake_openai(fail_n: int, exc: Exception, *, max_retries: int = 4) -> tuple[Client, _FlakyCompletions]:
    client = Client(
        base_url="http://example.invalid/v1",
        api_key="local",
        model="fake",
        max_retries=max_retries,
    )
    flaky = _FlakyCompletions(fail_n, exc)
    # Swap out the underlying AsyncOpenAI with a lightweight stand-in.
    client._client.chat = _Chat(flaky)  # type: ignore[attr-defined]
    client._client.models = _Models()  # type: ignore[attr-defined]
    return client, flaky
 
 
async def test_client_retries_transient_then_succeeds():
    err = httpx.ConnectError("boom")
    client, flaky = _client_with_fake_openai(fail_n=2, exc=err, max_retries=4)
    # Patch tenacity's default backoff to be instant in tests.
    original_wait = client_mod.wait_exponential
    client_mod.wait_exponential = lambda **_: wait_none()  # type: ignore[assignment]
    try:
        result = await client.complete([{"role": "user", "content": "hi"}], [], stream=False)
    finally:
        client_mod.wait_exponential = original_wait
        await client.aclose()
 
    assert flaky.calls == 3
    assert result.text == "ok"
 
 
async def test_client_raises_llmerror_after_retries_exhausted():
    err = httpx.ConnectError("perma-fail")
    client, flaky = _client_with_fake_openai(fail_n=10, exc=err, max_retries=2)
    original_wait = client_mod.wait_exponential
    client_mod.wait_exponential = lambda **_: wait_none()  # type: ignore[assignment]
    try:
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "hi"}], [], stream=False)
    finally:
        client_mod.wait_exponential = original_wait
        await client.aclose()
 
    assert flaky.calls == 2
 
 
async def test_parse_tool_call_tolerates_malformed_json():
    from squishy.client import _parse_tool_call
 
    tc = _parse_tool_call("c1", "read_file", "{not json")
    assert tc.name == "read_file"
    assert tc.args["_raw"] == "{not json"
    assert "invalid JSON" in tc.args["_tool_arg_error"]
 
    tc = _parse_tool_call("c2", "read_file", '"a string"')
    assert tc.args["_raw"] == '"a string"'
    assert "expected JSON object" in tc.args["_tool_arg_error"]


async def test_parse_xml_tool_calls_single():
    """Parse a single XML tool call from Qwen3-Coder format."""
    from squishy.client import _parse_xml_tool_calls

    text = (
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\nsetup.py\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    calls = _parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].args == {"path": "setup.py"}
    assert calls[0].id == "call_0"


async def test_parse_xml_tool_calls_multiple():
    """Parse multiple XML tool calls in one response."""
    from squishy.client import _parse_xml_tool_calls

    text = (
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>foo.py</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=edit_file>\n"
        "<parameter=path>bar.py</parameter>\n"
        "<parameter=old_str>x = 1</parameter>\n"
        "<parameter=new_str>x = 2</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = _parse_xml_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "read_file"
    assert calls[0].args["path"] == "foo.py"
    assert calls[1].name == "edit_file"
    assert calls[1].args["old_str"] == "x = 1"
    assert calls[1].args["new_str"] == "x = 2"


async def test_parse_xml_tool_calls_json_value():
    """JSON values in parameters are parsed as structured data."""
    from squishy.client import _parse_xml_tool_calls

    text = (
        '<tool_call>\n<function=run_command>\n'
        '<parameter=command>python test.py</parameter>\n'
        '<parameter=timeout>30</parameter>\n'
        '</function>\n</tool_call>'
    )
    calls = _parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].args["command"] == "python test.py"
    assert calls[0].args["timeout"] == 30  # parsed as int, not string


async def test_parse_xml_tool_calls_no_match():
    """Returns empty list when no XML tool calls present."""
    from squishy.client import _parse_xml_tool_calls

    assert _parse_xml_tool_calls("just plain text") == []
    assert _parse_xml_tool_calls("") == []


async def test_strip_xml_tool_calls():
    """Strips XML tool calls from text, preserving other content."""
    from squishy.client import _strip_xml_tool_calls

    text = (
        "I'll read that file for you.\n"
        "<tool_call>\n<function=read_file>\n"
        "<parameter=path>setup.py</parameter>\n"
        "</function>\n</tool_call>\n"
        "Let me know if you need more."
    )
    result = _strip_xml_tool_calls(text)
    assert "<tool_call>" not in result
    assert "I'll read that file for you." in result
    assert "Let me know if you need more." in result


async def test_dispatch_surfaces_tool_arg_error():
    """Dispatcher turns ``_tool_arg_error`` into an actionable error message."""
    import tempfile

    from squishy.tools import dispatch
    from squishy.tools.base import ToolContext

    with tempfile.TemporaryDirectory() as tmp:
        ctx = ToolContext(working_dir=tmp, permission_mode="yolo", use_sandbox=False)
        result = await dispatch(
            "read_file",
            {"_tool_arg_error": "invalid JSON: bad bracket", "_raw": "{bad"},
            ctx,
        )
    assert not result.success
    assert "invalid JSON" in result.error
