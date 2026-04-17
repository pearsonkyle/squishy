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
    assert tc.args == {"_raw": "{not json"}
 
    tc = _parse_tool_call("c2", "read_file", '"a string"')
    assert tc.args == {"_raw": '"a string"'}
