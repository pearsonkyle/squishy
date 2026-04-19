"""Async OpenAI-compatible client with retries, timeouts, streaming.
 
The only network-facing module. Uses openai.AsyncOpenAI under the hood.
Retry policy: exponential backoff on transient failures (timeout, connection,
5xx, rate-limit); immediate fail on auth/400/etc.
"""
 
from __future__ import annotations
 
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
 
import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
 
from squishy.errors import LLMError
 
log = logging.getLogger("squishy.client")
 
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    httpx.TimeoutException,
    httpx.ConnectError,
)
 
 
@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
 
 
@dataclass
class CompletionResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] | None = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return (self.usage or {}).get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return (self.usage or {}).get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return (self.usage or {}).get("total_tokens", 0)
 
 
OnTextFn = Callable[[str], Awaitable[None] | None]
 
 
@dataclass
class Client:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 8192
    request_timeout: float = 120.0
    max_retries: int = 4
    """Our own retry count. The underlying SDK retries are disabled to avoid double-counting."""
 
    _client: AsyncOpenAI = field(init=False, repr=False)
 
    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=0,
        )
 
    async def aclose(self) -> None:
        await self._client.close()
 
    async def __aenter__(self) -> Client:
        return self
 
    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception as e:  # noqa: BLE001
            log.debug("health check failed: %s", e)
            return False

    async def discover_model_name(self) -> str:
        """Try to discover the actual model name from the endpoint.
        
        Returns the configured model if discovery fails, or 'unknown-model'.
        """
        try:
            # Try to list models and get the first one
            models = await self._client.models.list()
            if models.data:
                # Return the first model's id
                return models.data[0].id
        except Exception:  # noqa: BLE001
            pass
        
        # Return configured model name if discovery fails
        return self.model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: OnTextFn | None = None,
    ) -> CompletionResult:
        """Run one chat completion. Retries transient failures with exponential backoff."""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type(TRANSIENT_ERRORS),
                reraise=True,
            ):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        log.warning(
                            "retry %d/%d after transient error",
                            attempt.retry_state.attempt_number,
                            self.max_retries,
                        )
                    if stream:
                        return await self._complete_stream(messages, tools, on_text)
                    return await self._complete_sync(messages, tools)
        except RetryError as e:  # pragma: no cover — AsyncRetrying with reraise=True
            raise LLMError(f"retries exhausted: {e}") from e
        except APIStatusError as e:
            raise LLMError(f"LLM returned {e.status_code}: {e.message}") from e
        except TRANSIENT_ERRORS as e:  # retries exhausted (reraise=True path)
            raise LLMError(f"transient error after {self.max_retries} retries: {e}") from e
 
        raise LLMError("unreachable")  # pragma: no cover
 
    async def _complete_sync(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> CompletionResult:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools or None,  # type: ignore[arg-type]
            tool_choice="auto" if tools else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append(_parse_tool_call(tc.id, tc.function.name, tc.function.arguments or "{}"))
        return CompletionResult(
            text=msg.content or "",
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
            usage=_usage_dict(resp.usage),
        )
 
    async def _complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: OnTextFn | None,
    ) -> CompletionResult:
        stream: AsyncIterator[Any] = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools or None,  # type: ignore[arg-type]
            tool_choice="auto" if tools else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
 
        text_parts: list[str] = []
        tc_buf: dict[int, dict[str, str]] = {}
        finish_reason = ""
        usage: dict[str, int] = {}

        async for chunk in stream:
            if not chunk.choices:
                # Some providers send a final chunk with usage but no choices.
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = _usage_dict(chunk_usage)
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                if on_text is not None:
                    result = on_text(content)
                    if result is not None:
                        await result
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = tc.index
                slot = tc_buf.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            # Capture usage from the final chunk if the provider includes it.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage_dict(chunk_usage)

        calls = [
            _parse_tool_call(
                tc_buf[i]["id"] or f"call_{i}",
                tc_buf[i]["name"],
                tc_buf[i]["arguments"] or "{}",
            )
            for i in sorted(tc_buf)
        ]
        return CompletionResult(
            text="".join(text_parts),
            tool_calls=calls,
            finish_reason=finish_reason,
            usage=usage,
        )
 
 
def _parse_tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            args = {"_raw": arguments}
    except json.JSONDecodeError:
        args = {"_raw": arguments}
    return ToolCall(id=call_id, name=name, args=args)
 
 
def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }