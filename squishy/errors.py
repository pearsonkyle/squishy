"""Typed exceptions. Boundary-checked at the API surface.

Rule of thumb: wrap network/subprocess errors at the seam, let domain errors
bubble. Callers pattern-match on these types; never on string messages.
"""

from __future__ import annotations


class SquishyError(Exception):
    """Base for all squishy errors."""


class LLMError(SquishyError):
    """LLM call failed after retries (connection, timeout, 5xx, rate-limit exhaustion)."""


class AgentTimeout(SquishyError):
    """Overall task wall-clock timeout exceeded."""


class AgentCancelled(SquishyError):
    """Task cancelled by the caller (e.g. Ctrl-C, asyncio.CancelledError)."""


class BenchError(SquishyError):
    """Benchmark harness setup/execution error."""
