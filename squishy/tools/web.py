"""Web fetch tool — retrieve documentation from a URL."""

from __future__ import annotations

import html
import re
from typing import Any

import httpx

from squishy.tools.base import Tool, ToolContext, ToolResult

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")
_TIMEOUT = 15.0
_MAX_BYTES = 200_000  # cap downloaded content


def _html_to_text(raw: str) -> str:
    """Minimal HTML-to-text: strip tags, decode entities, collapse whitespace."""
    # Remove script/style blocks entirely.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines.
    text = re.sub(r"<(br|p|div|li|tr|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags.
    text = _TAG_RE.sub("", text)
    # Decode HTML entities.
    text = html.unescape(text)
    # Collapse excessive newlines.
    text = _WS_RE.sub("\n\n", text).strip()
    return text


async def _fetch_url(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = args.get("url", "")
    if not url:
        return ToolResult(False, error="url is required")
    if not url.startswith(("http://", "https://")):
        return ToolResult(False, error="url must start with http:// or https://")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_TIMEOUT,
        ) as client:
            resp = await client.get(url, headers={"User-Agent": "squishy-agent/0.2"})
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return ToolResult(False, error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}")
    except httpx.TimeoutException:
        return ToolResult(False, error=f"Timed out fetching {url}")
    except httpx.RequestError as e:
        return ToolResult(False, error=f"Request failed: {e}")

    content_type = resp.headers.get("content-type", "")
    raw = resp.text[:_MAX_BYTES]

    if "html" in content_type:
        text = _html_to_text(raw)
    else:
        text = raw

    # Truncate to a reasonable size for the context window.
    max_chars = ctx.max_tool_output_chars if hasattr(ctx, "max_tool_output_chars") else 32_000
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated, {len(text) - max_chars} chars omitted ...]"

    return ToolResult(
        True,
        data={"url": str(resp.url), "content": text, "content_type": content_type},
        display=f"fetched {resp.url} ({len(text)} chars)",
    )


fetch_url = Tool(
    name="fetch_url",
    description=(
        "Fetch content from a URL (documentation, API references, etc.). "
        "HTML is converted to plain text. Use this to look up library docs, "
        "error messages, or reference material."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must start with http:// or https://)",
            },
        },
        "required": ["url"],
    },
    run=_fetch_url,
)

WEB_TOOLS: list[Tool] = [fetch_url]
