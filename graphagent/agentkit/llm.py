"""Model resolution, including OpenAI-compatible local endpoints.

The Agents SDK defaults to OpenAI's hosted Responses API. Pointing it at LM
Studio (or vLLM, or anything else that speaks ``/v1/chat/completions``) needs
three things, all of them easy to get wrong:

* an ``AsyncOpenAI`` client with the local ``base_url`` and a throwaway key,
* ``OpenAIChatCompletionsModel`` rather than the Responses-API default,
* tracing switched off, or every run tries to POST spans to api.openai.com
  with a key that doesn't exist.

A local endpoint will happily JIT-load whatever model id it is handed, so
there is deliberately **no default model name on the local path** -- omitting
``model`` raises instead of silently pulling something off disk.
"""

from __future__ import annotations

import os

from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from agents.models.interface import Model
from openai import AsyncOpenAI

DEFAULT_MODEL = "gpt-4.1"
"""Used only when talking to the hosted OpenAI API."""

_BASE_URL_ENV = ("GRAPHAGENT_BASE_URL", "SQUISHY_BASE_URL", "OPENAI_BASE_URL")


def discover_base_url(explicit: str | None = None) -> str | None:
    """First of ``explicit`` then the known env vars that is set."""
    if explicit:
        return explicit
    for name in _BASE_URL_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


def resolve_model(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str | Model:
    """Return something ``Agent(model=...)`` accepts.

    With no base URL this is just the model name and the SDK's own defaults
    apply. With one, it is a chat-completions model bound to a local client.
    """
    base_url = discover_base_url(base_url)
    if not base_url:
        return model or DEFAULT_MODEL
    if not model:
        raise ValueError(
            "a model id is required when base_url points at a local endpoint "
            f"({base_url}) -- it will load whatever id it is asked for"
        )
    set_tracing_disabled(True)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or "local",
    )
    return OpenAIChatCompletionsModel(model=model, openai_client=client)
