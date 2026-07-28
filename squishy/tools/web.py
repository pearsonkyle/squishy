"""Web fetch tool — retrieve documentation from a URL."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx

from squishy.tools.base import Tool, ToolContext, ToolResult

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")
_TIMEOUT = 15.0
_MAX_BYTES = 200_000  # cap downloaded content
_MAX_REDIRECTS = 5


def _is_blocked_ip(ip: str) -> bool:
    """True for private / loopback / link-local / reserved addresses.

    Blocks SSRF targets like the cloud-metadata endpoint (169.254.169.254),
    localhost, and internal RFC1918/ULA ranges.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → treat as unsafe
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _host_is_blocked(host: str) -> bool:
    """Resolve *host* and return True if any resolved address is non-public.

    Called for the initial URL and re-checked after every redirect so a
    public URL cannot 302 into an internal endpoint.
    """
    if not host:
        return True
    # A literal IP in the URL is checked directly (getaddrinfo would echo it).
    stripped = host.strip("[]")  # IPv6 literals arrive bracketed
    try:
        ipaddress.ip_address(stripped)
        return _is_blocked_ip(stripped)
    except ValueError:
        pass
    # Hostname: resolve every A/AAAA record; block if any is non-public
    # (defends against DNS returning both a public and an internal address).
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # cannot resolve → refuse
    for info in infos:
        ip = info[4][0]
        if _is_blocked_ip(ip):
            return True
    return False


def _ssrf_check(url: str) -> str | None:
    """Return an error string if *url* targets a non-public host, else None."""
    host = urlsplit(url).hostname or ""
    if _host_is_blocked(host):
        return (
            f"refused: {host or url!r} resolves to a private/loopback/link-local "
            "address (blocked to prevent SSRF to internal services)"
        )
    return None


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

    # SSRF guard on the initial URL before any request leaves the process.
    ssrf_err = _ssrf_check(url)
    if ssrf_err:
        return ToolResult(False, error=ssrf_err)

    content_type = ""
    final_url = url
    raw = ""
    try:
        # Redirects are followed manually so each hop can be SSRF-re-checked —
        # a public URL must not be able to 302 into an internal endpoint.
        async with httpx.AsyncClient(follow_redirects=False, timeout=_TIMEOUT) as client:
            current = url
            for _hop in range(_MAX_REDIRECTS + 1):
                async with client.stream(
                    "GET", current, headers={"User-Agent": "squishy-agent/0.2"},
                ) as resp:
                    if resp.is_redirect:
                        loc = resp.headers.get("location")
                        if not loc:
                            return ToolResult(False, error="redirect without a Location header")
                        nxt = str(resp.url.join(loc))
                        hop_err = _ssrf_check(nxt)
                        if hop_err:
                            return ToolResult(False, error=hop_err)
                        current = nxt
                        continue
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    final_url = str(resp.url)
                    # Read at most _MAX_BYTES so a huge response can't be
                    # buffered in full before we slice it.
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) >= _MAX_BYTES:
                            break
                    raw = bytes(buf[:_MAX_BYTES]).decode("utf-8", errors="replace")
                    break
            else:
                return ToolResult(False, error=f"too many redirects (>{_MAX_REDIRECTS})")
    except httpx.HTTPStatusError as e:
        return ToolResult(False, error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}")
    except httpx.TimeoutException:
        return ToolResult(False, error=f"Timed out fetching {url}")
    except httpx.RequestError as e:
        return ToolResult(False, error=f"Request failed: {e}")

    if "html" in content_type:
        text = _html_to_text(raw)
    else:
        text = raw

    # Truncate to a reasonable size for the context window.
    max_chars = getattr(ctx, "max_tool_output_chars", None) or 32_000
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated, {len(text) - max_chars} chars omitted ...]"

    return ToolResult(
        True,
        data={"url": final_url, "content": text, "content_type": content_type},
        display=f"fetched {final_url} ({len(text)} chars)",
    )


fetch_url = Tool(
    name="fetch_url",
    description="Fetch a URL and return its text (HTML stripped). For docs/reference lookups.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL"},
        },
        "required": ["url"],
    },
    run=_fetch_url,
)

WEB_TOOLS: list[Tool] = [fetch_url]
