"""Async OpenAI-compatible client with retries, timeouts, streaming.
 
The only network-facing module. Uses openai.AsyncOpenAI under the hood.
Retry policy: exponential backoff on transient failures (timeout, connection,
5xx, rate-limit); immediate fail on auth/400/etc.
"""
 
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
 
import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
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
    # A local LLM server (vLLM/LM Studio/llama.cpp) dropping the connection
    # mid-generation raises these — NOT ConnectError — while the SSE body is
    # being iterated. Without them here the disconnect is neither retried nor
    # translated to LLMError, and a raw httpx error escapes the client seam.
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
)


def _is_transient(exc: BaseException) -> bool:
    """Return True for transient errors that should be retried.

    Includes connection/timeout errors plus 5xx APIStatusError (server errors
    from vLLM restarts, OOM recovery, etc.).
    """
    if isinstance(exc, TRANSIENT_ERRORS):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False
 
 
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
    reasoning: str = ""  # Qwen3 thinking / chain-of-thought (separate from text)

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
# Fired when tenacity is about to sleep before retrying a transient error.
# Receives (attempt_number, max_attempts, exception) so the caller can
# both warn the user and reset any in-flight streaming state — without
# the reset, a partial first-attempt stream gets concatenated with the
# retry's full response and the user sees garbled output.
OnRetryFn = Callable[[int, int, BaseException], None]
 
 
@dataclass
class Client:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 8192
    request_timeout: float = 120.0
    max_retries: int = 8
    context_window: int = 0  # discovered from endpoint; 0 = unknown (no % display)
    thinking: bool = False
    """Our own retry count. The underlying SDK retries are disabled to avoid double-counting."""

    # Number of retries consumed by the most recent `complete()` call.
    # Reset at the top of each call, incremented in tenacity's before_sleep
    # callback.  The agent loop reads this to detect retry storms (vLLM
    # transient outages eating the per-task budget).
    last_call_retries: int = 0

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

    async def discover_model_name(self, *, timeout: float = 10.0) -> str:
        """Try to discover the actual model name from the endpoint.

        Bounded by a tight per-call timeout (default 10s) so a down
        endpoint can't strand the user on a 120s blank-screen wait
        before the REPL prompt appears. The configured model name is
        used as the fallback, and the failure is logged at WARNING
        so callers can surface it.

        As a side-effect, sets ``self.context_window`` when the endpoint
        exposes it (LM Studio returns ``context_length`` on model objects).
        """
        try:
            # Try to list models and get the first one
            models = await asyncio.wait_for(
                self._client.models.list(), timeout=timeout,
            )
            if models.data:
                model = models.data[0]
                # LM Studio (and some vLLM builds) expose context_length.
                ctx = getattr(model, "context_length", None)
                if isinstance(ctx, int) and ctx > 0:
                    self.context_window = ctx
                return model.id
        except TimeoutError:
            log.warning(
                "model discovery timed out after %.1fs against %s — "
                "endpoint may be down; using configured model name",
                timeout, self.base_url,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "model discovery failed against %s: %s — using configured model name",
                self.base_url, e,
            )

        # Return configured model name if discovery fails
        return self.model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = True,
        on_text: OnTextFn | None = None,
        on_retry: OnRetryFn | None = None,
    ) -> CompletionResult:
        """Run one chat completion. Retries transient failures with exponential backoff.

        ``on_retry`` (optional) is invoked just before each backoff sleep
        with ``(attempt_number, max_attempts, exception)``. The agent
        wires this to ``display.reset_streaming`` + a warn line so (a)
        the partial first-attempt stream is dropped before the retry
        appends to it, and (b) the user knows the wait isn't a hang.
        """
        # Reset per-call retry counter so the agent can read how many retries
        # this single completion consumed.
        self.last_call_retries = 0

        def _bump_retries(retry_state: Any) -> None:
            self.last_call_retries += 1
            if on_retry is not None:
                exc = None
                outcome = getattr(retry_state, "outcome", None)
                if outcome is not None:
                    try:
                        exc = outcome.exception()
                    except Exception:  # noqa: BLE001
                        exc = None
                try:
                    on_retry(
                        retry_state.attempt_number,
                        self.max_retries,
                        exc or RuntimeError("transient error"),
                    )
                except Exception:  # noqa: BLE001
                    # Never let a flaky display callback break the retry loop.
                    log.debug("on_retry callback failed", exc_info=True)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=60),
                retry=retry_if_exception(_is_transient),
                before_sleep=_bump_retries,
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
        except APIStatusError as e:
            # Azure / OpenAI content-filter rejections are 400s with
            # `content_filter` in the body.  These are deterministic — retrying
            # the same prompt will never succeed.  Surface them with a stable
            # marker so the agent loop can fail-fast on this instance instead
            # of burning the whole retry budget.
            msg = str(getattr(e, "message", "")) or str(e)
            msg_lower = msg.lower()
            if e.status_code == 400 and "content_filter" in msg_lower:
                raise LLMError(f"content_filter: {msg}") from e
            # Azure-strict schema violation: orphan tool_calls (assistant
            # message has tool_calls without paired tool responses), OR the
            # reverse — a role="tool" message not preceded by a matching
            # assistant tool_calls.  Same request will fail forever —
            # fail-fast instead of retrying.
            if e.status_code == 400 and (
                "tool_call_ids did not have response" in msg_lower
                or "must be followed by tool messages" in msg_lower
                or "must be a response to a preceding message" in msg_lower
                or ("role" in msg_lower and "tool" in msg_lower and "preceding" in msg_lower)
            ):
                raise LLMError(f"azure_strict_orphan_tool_calls: {msg}") from e
            raise LLMError(f"LLM returned {e.status_code}: {e.message}") from e
        except TRANSIENT_ERRORS as e:  # retries exhausted (reraise=True path)
            raise LLMError(f"transient error after {self.max_retries} retries: {e}") from e
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            # Catch-all so nothing untyped escapes the client seam — the
            # facade promises only squishy.errors types cross this boundary.
            # asyncio.CancelledError / KeyboardInterrupt are BaseException,
            # not Exception, so they still propagate for clean cancellation.
            raise LLMError(f"unexpected client error: {type(e).__name__}: {e}") from e
 
    def _build_create_kwargs(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # gpt-5 / o-series reasoning models reject `max_tokens` and require
        # `max_completion_tokens` instead.  Detect by model name prefix.
        m = (self.model or "").lower()
        is_reasoning_family = (
            m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")
            or m.startswith("o4")
        )
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            temperature=self.temperature,
        )
        if is_reasoning_family:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        if self.thinking:
            # Enable Qwen3 thinking mode — the model uses the `reasoning` field
            # for chain-of-thought, which improves tool-call formatting.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        return kwargs

    async def _complete_sync(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> CompletionResult:
        resp = await self._client.chat.completions.create(
            **self._build_create_kwargs(messages, tools),
            stream=False,
        )
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append(_parse_tool_call(tc.id, tc.function.name, tc.function.arguments or "{}"))
        # Capture reasoning separately for training data export.
        reasoning = getattr(msg, "reasoning", None) or ""
        # Fallback: some models (Qwen3 thinking mode) put text in `reasoning`.
        # Use `is None` check so an explicit empty-string content doesn't leak reasoning.
        text = msg.content if msg.content is not None else (reasoning or "")
        # Fallback: parse XML tool calls from text when server doesn't parse them.
        if not calls and text:
            xml_calls = _parse_xml_tool_calls(text)
            if xml_calls:
                calls = xml_calls
                text = _strip_xml_tool_calls(text)
                log.debug("parsed %d XML tool call(s) from text", len(calls))
        return CompletionResult(
            text=text,
            tool_calls=calls,
            finish_reason=choice.finish_reason or "",
            usage=_usage_dict(resp.usage),
            reasoning=reasoning,
        )
 
    async def _complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text: OnTextFn | None,
    ) -> CompletionResult:
        stream: AsyncIterator[Any] = await self._client.chat.completions.create(
            **self._build_create_kwargs(messages, tools),
            stream=True,
            stream_options={"include_usage": True},
        )
 
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
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
            # Capture reasoning separately for training data export.
            delta_reasoning = getattr(delta, "reasoning", None)
            if delta_reasoning:
                reasoning_parts.append(delta_reasoning)
            raw_content = getattr(delta, "content", None)
            if raw_content is not None:
                text_parts.append(raw_content)
                if on_text is not None:
                    result = on_text(raw_content)
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
        text = "".join(text_parts)
        reasoning = "".join(reasoning_parts)
        # Fallback: use reasoning as text when no content was streamed
        # (mirrors sync path: msg.content is None -> use reasoning).
        if not text and reasoning:
            text = reasoning
        # Fallback: parse XML tool calls from streamed text when server doesn't.
        if not calls and text:
            xml_calls = _parse_xml_tool_calls(text)
            if xml_calls:
                calls = xml_calls
                text = _strip_xml_tool_calls(text)
                log.debug("parsed %d XML tool call(s) from stream", len(calls))
        return CompletionResult(
            text=text,
            tool_calls=calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning=reasoning,
        )
 
 
# Regex for Qwen3-style XML tool calls:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>VALUE</parameter>
#   ...
#   </function>
#   </tool_call>
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)


def _parse_xml_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from XML-formatted text (Qwen3-Coder fallback).

    Returns an empty list if no XML tool calls are found.
    """
    calls: list[ToolCall] = []
    for i, m in enumerate(_XML_TOOL_CALL_RE.finditer(text)):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}
        for pm in _XML_PARAM_RE.finditer(body):
            key = pm.group(1)
            raw = pm.group(2).strip()
            # Try to parse as JSON for structured values (lists, dicts, numbers).
            try:
                args[key] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                args[key] = raw
        calls.append(ToolCall(id=f"call_{i}", name=name, args=args))
    return calls


def _strip_xml_tool_calls(text: str) -> str:
    """Remove XML tool call blocks from text, returning remaining content."""
    return _XML_TOOL_CALL_RE.sub("", text).strip()


def _parse_tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    try:
        args = json.loads(arguments)
        if not isinstance(args, dict):
            args = {
                "_tool_arg_error": f"expected JSON object, got {type(args).__name__}",
                "_raw": arguments[:800],
            }
    except json.JSONDecodeError as e:
        # Surface a clear, actionable error so the tool dispatcher returns a
        # message the model can correct, instead of a silent "_raw" payload
        # that looks like a missing-required-field error downstream.
        args = {
            "_tool_arg_error": f"invalid JSON in tool arguments: {e}",
            "_raw": arguments[:800],
        }
    return ToolCall(id=call_id, name=name, args=args)
 
 
def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }