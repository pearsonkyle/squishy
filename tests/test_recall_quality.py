"""Recall ranking quality: a regression harness for the lexical scorer.

Scoring changes are easy to make and hard to trust — a tweak that fixes one
query silently ruins three others. These cases pin the behaviour we care about
(the right file ranked first, few results, bounded tokens) against a fixture
repo shaped like a real project, so future tuning has a ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squishy.index import build_index, save_index
from squishy.tools.base import ToolContext
from squishy.tools.recall import _recall


def _mkrepo(root: Path) -> None:
    """A repo with deliberate lexical traps: many files share common tokens
    (handler/service/manager), and the distinguishing token is rare."""
    (root / "auth").mkdir()
    (root / "auth" / "session.py").write_text(
        '"""Session token issuing and validation."""\n'
        "class SessionManager:\n"
        "    def issue_token(self, user): ...\n"
        "    def revoke_token(self, token): ...\n"
    )
    (root / "auth" / "password.py").write_text(
        '"""Password hashing helpers."""\n'
        "def hash_password(raw): ...\n"
        "def verify_password(raw, digest): ...\n"
    )
    (root / "billing").mkdir()
    (root / "billing" / "invoice.py").write_text(
        '"""Invoice generation and PDF rendering."""\n'
        "class InvoiceManager:\n"
        "    def render_pdf(self, invoice): ...\n"
    )
    (root / "billing" / "handler.py").write_text(
        '"""Billing webhook handler."""\n'
        "class BillingHandler:\n"
        "    def handle(self, event): ...\n"
    )
    (root / "api").mkdir()
    (root / "api" / "handler.py").write_text(
        '"""Generic request handler base."""\n'
        "class RequestHandler:\n"
        "    def handle(self, request): ...\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_session.py").write_text(
        '"""Tests for the session manager."""\n'
        "def test_issue_token(): ...\n"
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _mkrepo(tmp_path)
    save_index(str(tmp_path), build_index(str(tmp_path)))
    return tmp_path


def _ctx(repo: Path) -> ToolContext:
    return ToolContext(working_dir=str(repo), permission_mode="plan", use_sandbox=False)


async def _top_paths(repo: Path, query: str, **kw) -> list[str]:
    r = await _recall({"query": query, **kw}, _ctx(repo))
    assert r.success, r.error
    return [e["path"] for e in r.data["results"]]


# -- ranking ---------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("revoke_token", "auth/session.py"),
        ("hash password", "auth/password.py"),
        ("render invoice pdf", "billing/invoice.py"),
        ("billing webhook", "billing/handler.py"),
    ],
)
async def test_top_hit_is_the_right_file(repo: Path, query: str, expected: str) -> None:
    paths = await _top_paths(repo, query)
    assert paths, f"no results for {query!r}"
    assert paths[0] == expected, f"{query!r} -> {paths[:3]}"


async def test_natural_language_query_still_finds_the_file(repo: Path) -> None:
    """Filler words must not break a plain-English question. (Filtering query
    stopwords was tried and measurably WORSENED ranking, so the scorer keeps
    every token — this pins that a wordy query still works.)"""
    paths = await _top_paths(repo, "how does the code that revokes a token work")
    assert paths and paths[0] == "auth/session.py", paths[:3]


async def test_source_outranks_test_file(repo: Path) -> None:
    paths = await _top_paths(repo, "session manager")
    assert paths[0] == "auth/session.py"
    if "tests/test_session.py" in paths:
        assert paths.index("auth/session.py") < paths.index("tests/test_session.py")


# -- result size / budget ---------------------------------------------------

async def test_token_budget_is_respected(repo: Path) -> None:
    r = await _recall({"query": "handler", "token_budget": 60, "depth": 2}, _ctx(repo))
    assert r.success
    size_tokens = len(json.dumps(r.data["results"], ensure_ascii=False)) / 3.5
    # The first hit is always emitted, so allow one entry's overshoot.
    assert size_tokens < 400, size_tokens
    if r.data["returned"] < r.data["total_matched"]:
        assert "truncated" in r.data


async def test_truncation_is_announced(repo: Path) -> None:
    r = await _recall({"query": "handler", "token_budget": 1, "depth": 2}, _ctx(repo))
    assert r.success
    if r.data["returned"] < r.data["total_matched"]:
        assert "budget" in r.data["truncated"]
