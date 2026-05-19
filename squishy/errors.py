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
    """Overall task wall-clock timeout exceeded.

    May carry a ``partial_result`` attribute (TaskResult) built from the
    in-progress LoopState so callers (the bench harness) can recover the
    transcript / turn_log accumulated up to the timeout.
    """

    partial_result: object | None = None  # TaskResult; loose typing avoids cycle


class AgentCancelled(SquishyError):
    """Task cancelled by the caller (e.g. Ctrl-C, asyncio.CancelledError).

    May carry a ``partial_result`` like AgentTimeout.
    """

    partial_result: object | None = None


class BenchError(SquishyError):
    """Benchmark harness setup/execution error."""
