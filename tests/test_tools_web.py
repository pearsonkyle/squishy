"""Tests for the web fetch tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx

from squishy.tools.web import _html_to_text, fetch_url


async def test_fetch_url_success(ctx):
    """fetch_url should return content from a successful HTTP response."""
    mock_resp = httpx.Response(
        200,
        text="<html><body><h1>Hello</h1><p>World</p></body></html>",
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://example.com/docs"),
    )

    with patch("squishy.tools.web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

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
    mock_resp = httpx.Response(
        404,
        text="Not Found",
        headers={"content-type": "text/plain"},
        request=httpx.Request("GET", "https://example.com/missing"),
    )

    with patch("squishy.tools.web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        r = await fetch_url.run({"url": "https://example.com/missing"}, ctx)

    assert not r.success
    assert "404" in r.error


async def test_fetch_url_timeout(ctx):
    with patch("squishy.tools.web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        r = await fetch_url.run({"url": "https://example.com/slow"}, ctx)

    assert not r.success
    assert "Timed out" in r.error


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


async def test_fetch_url_plain_text(ctx):
    """Non-HTML content should be returned as-is."""
    mock_resp = httpx.Response(
        200,
        text="plain text content",
        headers={"content-type": "text/plain"},
        request=httpx.Request("GET", "https://example.com/file.txt"),
    )

    with patch("squishy.tools.web.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        r = await fetch_url.run({"url": "https://example.com/file.txt"}, ctx)

    assert r.success
    assert r.data["content"] == "plain text content"
