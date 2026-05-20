"""ACP (Agent Client Protocol) integration for squishy.

Lets editors that speak ACP — Zed, Neovim, and others — drive squishy as a
backend coding agent. Squishy runs as the *agent* side of the protocol; the
editor is the client.

Entry point: ``squishy-acp`` (defined in pyproject.toml), or
``squishy.acp.cli:run``.

The integration is purely additive: it reuses ``squishy.agent.Agent`` (one
per ACP session) and bridges its existing hooks — ``Display`` →
``session/update``, ``prompt_fn`` → ``session/request_permission``,
``ToolContext.fs_client`` → ``fs/read_text_file``+``fs/write_text_file``,
``ToolContext.terminal_client`` → ``terminal/*``. The legacy CLI is
untouched.
"""
from __future__ import annotations

from squishy.acp.agent import SquishyAcpAgent

__all__ = ["SquishyAcpAgent"]
