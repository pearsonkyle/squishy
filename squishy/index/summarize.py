"""LLM summaries for index nodes.

Deterministic-first: files that already ship with a docstring or header
comment skip the LLM entirely. Everything else is batched with a bounded
`asyncio.Semaphore` and capped by `max_tokens` budget so a large repo can't
silently burn the model's context window.

Dir-level summaries roll up their children with one compact call each.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Protocol

from squishy.index.model import Index, Node

DEFAULT_CONCURRENCY = 4
DEFAULT_TOKEN_BUDGET = 100_000
FILE_SUMMARY_MAX_TOKENS = 120
DIR_SUMMARY_MAX_TOKENS = 80
FILE_EXCERPT_LINES = 200  # cap how much of a file we send to the LLM
 
 
FILE_PROMPT = (
    "Summarize this source file in ONE sentence: what does it do, and what "
    "are its 2-3 key exports? Plain text, no preamble."
)
DIR_PROMPT = (
    "Given these file summaries from a directory, write ONE sentence describing "
    "what the directory is for. Plain text, no preamble."
)
 
 
class ClientLike(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool = ...,
    ) -> Any: ...
 
 
@dataclass
class ProgressEvent:
    done: int
    total: int
    label: str = ""
 
 
ProgressFn = Any  # Callable[[ProgressEvent], None] | None — kept loose for cheap import
 
 
def _excerpt(abs_path: str, max_lines: int = FILE_EXCERPT_LINES) -> str:
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append("...<truncated>")
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except OSError:
        return ""
 
 
def _first_line(s: str) -> str:
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""
 
 
@dataclass
class Summarizer:
    client: ClientLike
    cwd: str
    concurrency: int = DEFAULT_CONCURRENCY
    token_budget: int = DEFAULT_TOKEN_BUDGET
    model_name: str = ""
    _tokens_used: int = 0
    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _budget_exhausted: bool = False
    _files_summarized: int = 0
    _dirs_summarized: int = 0
 
    async def summarize(
        self,
        index: Index,
        *,
        progress: Any = None,
    ) -> None:
        """Summarize file and dir nodes in-place.

        Files whose `summary` is already non-empty (docstring / header comment
        / reused from prior index) are skipped.

        Directory summaries use hash-based caching: if children's summaries
        haven't changed, the dir summary is reused.
        """
        files_needing = [n for n in index.root.walk() if n.kind == "file" and not n.summary]
        total = len(files_needing)
        sem = asyncio.Semaphore(max(1, self.concurrency))
        done = 0
        lock = asyncio.Lock()

        async def _one(node: Node) -> None:
            nonlocal done
            async with sem:
                if self._budget_exhausted:
                    return
                await self._summarize_file(node)
                async with lock:
                    done += 1
                    if progress is not None:
                        with contextlib.suppress(Exception):
                            progress(ProgressEvent(done=done, total=total, label=node.path))

        if files_needing:
            await asyncio.gather(*[_one(n) for n in files_needing])

        # Dir rollups after files are settled.
        await self._summarize_dirs(index.root)
        
        # Update meta with summary stats
        index.meta.summary_stats = {
            "files_summarized": self._files_summarized,
            "dirs_summarized": self._dirs_summarized,
        }
 
    async def _summarize_file(self, node: Node) -> None:
        abs_path = os.path.join(self.cwd, node.path)
        text = _excerpt(abs_path)
        if not text.strip():
            return
        prompt = (
            f"{FILE_PROMPT}\n\n"
            f"# path: {node.path}\n"
            f"```\n{text}\n```"
        )
        summary = await self._ask(prompt, FILE_SUMMARY_MAX_TOKENS)
        if summary:
            node.summary = _first_line(summary)[:240]
            self._files_summarized += 1
 
    async def _summarize_dirs(self, node: Node) -> None:
        if node.kind not in ("repo", "dir"):
            return
        for child in node.children:
            await self._summarize_dirs(child)
        
        # Compute summary hash from children to detect changes
        child_summary_hashes = []
        for c in node.children:
            if c.kind in ("file", "dir"):
                child_summary_hashes.append(f"{c.path}:{c.summary_hash or c.summary}")
        
        current_hash = hashlib.md5(
            "|".join(sorted(child_summary_hashes)).encode()
        ).hexdigest()[:16]
        
        # Reuse existing dir summary if children haven't changed
        if node.summary and current_hash == node.summary_hash:
            return
        
        # Update hash and regenerate summary if needed
        node.summary_hash = current_hash
        
        if node.summary or self._budget_exhausted:
            return
        child_summaries = []
        for c in node.children[:8]:
            if c.summary:
                child_summaries.append(f"- {c.name}: {c.summary}")
            elif c.kind in ("file", "dir"):
                child_summaries.append(f"- {c.name}")
        if len(child_summaries) < 2:
            return
        prompt = (
            f"{DIR_PROMPT}\n\n"
            f"# path: {node.path or '.'}\n"
            + "\n".join(child_summaries)
        )
        summary = await self._ask(prompt, DIR_SUMMARY_MAX_TOKENS)
        if summary:
            node.summary = _first_line(summary)[:240]
            self._dirs_summarized += 1
 
    async def _ask(self, prompt: str, max_tokens: int) -> str:
        if self._budget_exhausted:
            return ""
        messages = [
            {"role": "system", "content": "You write terse one-sentence code summaries."},
            {"role": "user", "content": prompt},
        ]
        try:
            res = await self.client.complete(messages, [], stream=False)
        except Exception:  # noqa: BLE001
            # If one summary fails, don't torpedo the whole index.
            return ""
        usage = getattr(res, "usage", {}) or {}
        prompt_tok = int(usage.get("prompt_tokens", 0) or 0)
        completion_tok = int(usage.get("completion_tokens", 0) or 0)
        
        # If only total_tokens is provided (old-style API), split it
        if not prompt_tok and not completion_tok:
            total = int(usage.get("total_tokens", 0) or 0)
            prompt_tok, completion_tok = total // 2, total - (total // 2)
        
        self._prompt_tokens += prompt_tok
        self._completion_tokens += completion_tok
        self._tokens_used = self._prompt_tokens + self._completion_tokens
        if self.token_budget and self._tokens_used >= self.token_budget:
            self._budget_exhausted = True
        return getattr(res, "text", "") or ""
 
 
__all__ = [
    "Summarizer",
    "ProgressEvent",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_TOKEN_BUDGET",
]
