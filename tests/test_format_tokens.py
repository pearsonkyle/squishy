"""Tests for display.format_tokens_k."""

from __future__ import annotations

from squishy.display import format_tokens_k


def test_below_1k_no_window() -> None:
    assert format_tokens_k(0) == "0"
    assert format_tokens_k(847) == "847"


def test_below_1k_with_window() -> None:
    assert format_tokens_k(0, 128_000) == "0 (0%)"
    assert format_tokens_k(500, 128_000) == "500 (0%)"


def test_one_k_exact() -> None:
    assert format_tokens_k(1000, 128_000) == "1.0 K (1%)"


def test_k_with_decimal() -> None:
    assert format_tokens_k(1234, 0) == "1.2 K"
    assert format_tokens_k(12345, 128_000) == "12.3 K (10%)"


def test_percent_rounds() -> None:
    # 3840 / 128000 = 3.0% exactly
    assert format_tokens_k(3840, 128_000) == "3.8 K (3%)"
    # 6400 / 128000 = 5% exactly
    assert format_tokens_k(6400, 128_000) == "6.4 K (5%)"
    # 25600 / 128000 = 20% exactly
    assert format_tokens_k(25_600, 128_000) == "25.6 K (20%)"


def test_percent_above_100() -> None:
    assert format_tokens_k(200_000, 100_000) == "200.0 K (200%)"


def test_window_zero_suppresses_percent() -> None:
    assert format_tokens_k(12_345, 0) == "12.3 K"
    assert format_tokens_k(12_345, -1) == "12.3 K"
