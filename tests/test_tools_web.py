"""Tests for the web fetch tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from squishy.tools.web import _html_to_text, _is_blocked_ip, fetch_url


class _FakeResp:
    """Minimal stand-in for a streamed httpx.Response."""

    def __init__(self, status, *, content=b"", headers=None, url="https://example.com/docs"):
        self.status_code = status
        self.headers = httpx.Headers(headers or {})
        self.url = httpx.URL(url)
        self._content = content

    @property
    def is_redirect(self) -> bool:
        return 300 <= self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("GET", self.url)
            raise httpx.HTTPStatusError(
                "err", request=req, response=httpx.Response(self.status_code, request=req),
            )

    async def aiter_bytes(self):
        yield self._content


class _StreamCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *_a):
        return False


def _mock_client(*, stream_return=None, stream_side_effect=None):
    client = MagicMock()
    client.__aenter__ = _amock(client)
    client.__aexit__ = _amock(False)
    if stream_side_effect is not None:
        client.stream = MagicMock(side_effect=stream_side_effect)
    else:
        client.stream = MagicMock(return_value=_StreamCM(stream_return))
    return client


def _amock(retval):
    async def _inner(*_a, **_k):
        return retval
    return _inner


async def test_fetch_url_success(ctx):
    resp = _FakeResp(
        200,
        content=b"<html><body><h1>Hello</h1><p>World</p></body></html>",
        headers={"content-type": "text/html"},
    )
    with patch("squishy.tools.web._host_is_blocked", return_value=False), \
            patch("squishy.tools.web.httpx.AsyncClient", return_value=_mock_client(stream_return=resp)):
        r = await fetch_url.run({"url": "https://example.com/docs"}, ctx)
    assert r.success
    assert "Hello" in r.data["content"]
    assert "World" in r.data["content"]


async def test_fetch_url_missing_url(ctx):
    r = await fetch_url.run({}, ctx)
    assert not r.success
    assert "url is required" in r.error


async def test_fetch_url_invalid_scheme(ctx):
    r = await fetch_url.run({"url": "ftp://example.com"}, ctx)
    assert not r.success
    assert "http" in r.error


async def test_fetch_url_http_error(ctx):
    resp = _FakeResp(404, headers={"content-type": "text/plain"}, url="https://example.com/missing")
    with patch("squishy.tools.web._host_is_blocked", return_value=False), \
            patch("squishy.tools.web.httpx.AsyncClient", return_value=_mock_client(stream_return=resp)):
        r = await fetch_url.run({"url": "https://example.com/missing"}, ctx)
    assert not r.success
    assert "404" in r.error


async def test_fetch_url_timeout(ctx):
    with patch("squishy.tools.web._host_is_blocked", return_value=False), \
            patch(
                "squishy.tools.web.httpx.AsyncClient",
                return_value=_mock_client(stream_side_effect=httpx.TimeoutException("timed out")),
            ):
        r = await fetch_url.run({"url": "https://example.com/slow"}, ctx)
    assert not r.success
    assert "Timed out" in r.error


async def test_fetch_url_plain_text(ctx):
    resp = _FakeResp(
        200, content=b"plain text content",
        headers={"content-type": "text/plain"}, url="https://example.com/file.txt",
    )
    with patch("squishy.tools.web._host_is_blocked", return_value=False), \
            patch("squishy.tools.web.httpx.AsyncClient", return_value=_mock_client(stream_return=resp)):
        r = await fetch_url.run({"url": "https://example.com/file.txt"}, ctx)
    assert r.success
    assert r.data["content"] == "plain text content"


# ---- SSRF guard ----------------------------------------------------------

async def test_fetch_url_blocks_metadata_ip(ctx):
    """A literal cloud-metadata / link-local IP is refused (no DNS needed)."""
    r = await fetch_url.run({"url": "http://169.254.169.254/latest/meta-data/"}, ctx)
    assert not r.success
    assert "SSRF" in r.error or "private" in r.error


async def test_fetch_url_blocks_localhost_ip(ctx):
    r = await fetch_url.run({"url": "http://127.0.0.1:8080/admin"}, ctx)
    assert not r.success


async def test_fetch_url_blocks_redirect_to_internal(ctx):
    """A public URL that redirects to an internal address is stopped at the hop."""
    redirect = _FakeResp(
        302,
        headers={"location": "http://169.254.169.254/latest/meta-data/"},
        url="https://example.com/go",
    )
    # Resolve example.com to a public IP (offline-safe); the real _is_blocked_ip
    # then blocks the literal link-local redirect target.
    def _fake_getaddrinfo(host, *_a, **_k):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]
    with patch("squishy.tools.web.socket.getaddrinfo", side_effect=_fake_getaddrinfo), \
            patch("squishy.tools.web.httpx.AsyncClient", return_value=_mock_client(stream_return=redirect)):
        r = await fetch_url.run({"url": "https://example.com/go"}, ctx)
    assert not r.success
    assert "SSRF" in r.error or "private" in r.error


def test_is_blocked_ip_ranges():
    assert _is_blocked_ip("169.254.169.254")  # link-local (cloud metadata)
    assert _is_blocked_ip("127.0.0.1")        # loopback
    assert _is_blocked_ip("10.0.0.5")         # private
    assert _is_blocked_ip("192.168.1.1")      # private
    assert _is_blocked_ip("::1")              # IPv6 loopback
    assert _is_blocked_ip("fd00::1")          # IPv6 ULA
    assert not _is_blocked_ip("93.184.216.34")  # public (example.com)
    assert not _is_blocked_ip("8.8.8.8")        # public


# ---- HTML → text ---------------------------------------------------------

def test_html_to_text_strips_tags():
    assert "Hello" in _html_to_text("<b>Hello</b>")
    assert "<b>" not in _html_to_text("<b>Hello</b>")


def test_html_to_text_strips_script():
    result = _html_to_text("<script>alert('xss')</script><p>Safe</p>")
    assert "alert" not in result
    assert "Safe" in result


def test_html_to_text_decodes_entities():
    assert "&" in _html_to_text("&amp;")
    assert "<" in _html_to_text("&lt;")
