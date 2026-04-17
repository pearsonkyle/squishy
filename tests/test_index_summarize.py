"""Summarizer: skips files with existing summaries, respects token budget."""
 
from __future__ import annotations
 
from dataclasses import dataclass
from pathlib import Path
from typing import Any
 
import pytest
 
from squishy.index import build_index
from squishy.index.summarize import Summarizer
 
pytestmark = pytest.mark.asyncio
 
 
@dataclass
class _FakeResp:
    text: str = "ok"
    usage: dict[str, int] | None = None
 
 
class _FakeClient:
    def __init__(self, *, tokens_per_call: int = 100) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self._tokens = tokens_per_call
 
    async def complete(self, messages, tools, *, stream: bool = False):
        self.calls.append(messages)
        label = f"call{len(self.calls)}"
        return _FakeResp(text=f"sum {label}", usage={"total_tokens": self._tokens})
 
 
def _make_repo(root: Path) -> None:
    (root / "with_doc.py").write_text('"""Already has a summary."""\ndef f(): pass\n')
    (root / "no_doc.py").write_text("def g(): return 1\n")
    (root / "tiny.js").write_text("function hi() {}\n")
 
 
async def test_skips_files_with_existing_summary(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    idx = build_index(str(tmp_path))
    client = _FakeClient()
    summarizer = Summarizer(client=client, cwd=str(tmp_path), concurrency=2, token_budget=0)
    await summarizer.summarize(idx)
 
    with_doc = idx.find_file("with_doc.py")
    assert with_doc is not None
    assert with_doc.summary == "Already has a summary."  # unchanged
 
    no_doc = idx.find_file("no_doc.py")
    assert no_doc is not None
    assert no_doc.summary.startswith("sum call")
 
 
async def test_budget_halts_calls(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text(f"def x{i}(): return {i}\n")
    idx = build_index(str(tmp_path))
    # Budget = 150; first call consumes 100 → budget exhausted after first response.
    client = _FakeClient(tokens_per_call=100)
    summarizer = Summarizer(client=client, cwd=str(tmp_path), concurrency=1, token_budget=150)
    await summarizer.summarize(idx)
    # Concurrency=1 + budget exhausted after first call means we make 1-2 LLM calls
    # at most (second could start before the first finishes). Must be << 5.
    assert len(client.calls) < 5
 
 
async def test_client_failure_does_not_crash(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a(): pass\n")
    idx = build_index(str(tmp_path))
 
    class _Boom:
        async def complete(self, *_a, **_k):
            raise RuntimeError("nope")
 
    summarizer = Summarizer(client=_Boom(), cwd=str(tmp_path), concurrency=1, token_budget=0)
    await summarizer.summarize(idx)  # must not raise
    a = idx.find_file("a.py")
    assert a is not None
    # No summary added, but structure preserved.
    assert a.summary == ""
